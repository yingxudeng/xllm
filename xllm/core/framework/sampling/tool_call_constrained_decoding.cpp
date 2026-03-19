/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tool_call_constrained_decoding.h"

#include <dlpack/dlpack.h>
#include <glog/logging.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cctype>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "util/tensor_helper.h"

namespace xllm {

struct ToolCallTokenCache {
  std::vector<std::string> token_text_cache;
  std::array<std::vector<int32_t>, 256> token_ids_by_first_byte;
  std::shared_ptr<xgrammar::TokenizerInfo> xgrammar_tokenizer_info;
  std::shared_ptr<xgrammar::GrammarCompiler> xgrammar_compiler;
};

class JsonSchemaCursor {
 public:
  virtual ~JsonSchemaCursor() = default;

  virtual void advance(
      unsigned char c,
      std::vector<std::shared_ptr<const JsonSchemaCursor>>& out) const = 0;

  virtual bool is_complete() const = 0;

  virtual void collect_next_bytes(std::array<bool, 256>& mask) const = 0;

  virtual bool is_open_ended_text_frontier() const { return false; }
};

namespace {

using CursorPtr = std::shared_ptr<const JsonSchemaCursor>;
using CursorVec = std::vector<CursorPtr>;

struct TokenCacheKey {
  const Tokenizer* tokenizer;
  int32_t vocab_size;

  bool operator==(const TokenCacheKey& other) const {
    return tokenizer == other.tokenizer && vocab_size == other.vocab_size;
  }
};

struct TokenCacheKeyHash {
  size_t operator()(const TokenCacheKey& key) const {
    return std::hash<const Tokenizer*>{}(key.tokenizer) ^
           (std::hash<int32_t>{}(key.vocab_size) << 1);
  }
};

std::shared_ptr<const ToolCallTokenCache> get_or_build_token_cache(
    const Tokenizer& tokenizer,
    int32_t vocab_size) {
  static std::mutex cache_mutex;
  static std::unordered_map<TokenCacheKey,
                            std::weak_ptr<const ToolCallTokenCache>,
                            TokenCacheKeyHash>
      cache_by_key;

  const TokenCacheKey key{&tokenizer, vocab_size};
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache_by_key.find(key);
    if (it != cache_by_key.end()) {
      if (auto cached = it->second.lock()) {
        return cached;
      }
    }
  }

  auto built = std::make_shared<ToolCallTokenCache>();
  built->token_text_cache.resize(vocab_size);
  for (auto& bucket : built->token_ids_by_first_byte) {
    bucket.clear();
  }
  for (int32_t token_id = 0; token_id < vocab_size; ++token_id) {
    auto text = tokenizer.id_to_token(token_id);
    if (text.empty()) {
      std::vector<int32_t> single = {token_id};
      text = tokenizer.decode(Slice<int32_t>(single.data(), single.size()),
                              /*skip_special_tokens=*/false);
    }
    built->token_text_cache[token_id] = std::move(text);
    if (!built->token_text_cache[token_id].empty()) {
      const auto first =
          static_cast<unsigned char>(built->token_text_cache[token_id].front());
      built->token_ids_by_first_byte[first].push_back(token_id);
    }
  }

  try {
    built->xgrammar_tokenizer_info = std::make_shared<xgrammar::TokenizerInfo>(
        built->token_text_cache, xgrammar::VocabType::RAW, vocab_size);
    built->xgrammar_compiler = std::make_shared<xgrammar::GrammarCompiler>(
        *built->xgrammar_tokenizer_info);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to initialize xgrammar tokenizer/compiler cache: "
                 << e.what();
    built->xgrammar_tokenizer_info.reset();
    built->xgrammar_compiler.reset();
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& slot = cache_by_key[key];
  if (auto cached = slot.lock()) {
    return cached;
  }
  slot = built;
  return built;
}

void add_byte(std::array<bool, 256>& mask, unsigned char c) { mask[c] = true; }

void add_whitespace(std::array<bool, 256>& mask) {
  add_byte(mask, ' ');
  add_byte(mask, '\n');
  add_byte(mask, '\r');
  add_byte(mask, '\t');
}

bool is_json_whitespace(unsigned char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

std::string json_dump(const nlohmann::json& value) {
  return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::ignore);
}

bool schema_allows_type(const nlohmann::json& schema, const std::string& type) {
  if (!schema.is_object() || !schema.contains("type")) {
    return false;
  }
  const auto& type_value = schema["type"];
  if (type_value.is_string()) {
    return type_value.get<std::string>() == type;
  }
  if (type_value.is_array()) {
    return std::any_of(
        type_value.begin(), type_value.end(), [&](const nlohmann::json& item) {
          return item.is_string() && item.get<std::string>() == type;
        });
  }
  return false;
}

class AnyOfCursor final : public JsonSchemaCursor {
 public:
  explicit AnyOfCursor(CursorVec choices) : choices_(std::move(choices)) {}

  void advance(unsigned char c, CursorVec& out) const override {
    for (const auto& choice : choices_) {
      choice->advance(c, out);
    }
  }

  bool is_complete() const override {
    return std::any_of(choices_.begin(),
                       choices_.end(),
                       [](const CursorPtr& c) { return c->is_complete(); });
  }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    for (const auto& choice : choices_) {
      choice->collect_next_bytes(mask);
    }
  }

  bool is_open_ended_text_frontier() const override {
    return !choices_.empty() &&
           std::all_of(
               choices_.begin(), choices_.end(), [](const CursorPtr& c) {
                 return c->is_open_ended_text_frontier();
               });
  }

 private:
  CursorVec choices_;
};

class AllOfCursor final : public JsonSchemaCursor {
 public:
  explicit AllOfCursor(CursorVec branches) : branches_(std::move(branches)) {}

  void advance(unsigned char c, CursorVec& out) const override {
    std::vector<CursorVec> branch_outputs;
    branch_outputs.reserve(branches_.size());
    for (const auto& branch : branches_) {
      CursorVec next;
      branch->advance(c, next);
      if (next.empty()) {
        return;
      }
      branch_outputs.push_back(std::move(next));
    }
    CursorVec combination;
    combination.reserve(branch_outputs.size());
    expand_combinations(branch_outputs, 0, combination, out);
  }

  bool is_complete() const override {
    return std::all_of(branches_.begin(),
                       branches_.end(),
                       [](const CursorPtr& c) { return c->is_complete(); });
  }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    mask.fill(true);
    std::array<bool, 256> branch_mask{};
    for (const auto& branch : branches_) {
      branch_mask.fill(false);
      branch->collect_next_bytes(branch_mask);
      for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = mask[i] && branch_mask[i];
      }
    }
  }

  bool is_open_ended_text_frontier() const override {
    return !branches_.empty() &&
           std::all_of(
               branches_.begin(), branches_.end(), [](const CursorPtr& c) {
                 return c->is_open_ended_text_frontier();
               });
  }

 private:
  static void expand_combinations(const std::vector<CursorVec>& branch_outputs,
                                  size_t index,
                                  CursorVec& current,
                                  CursorVec& out) {
    if (index == branch_outputs.size()) {
      out.push_back(std::make_shared<AllOfCursor>(current));
      return;
    }
    for (const auto& state : branch_outputs[index]) {
      current.push_back(state);
      expand_combinations(branch_outputs, index + 1, current, out);
      current.pop_back();
    }
  }

  CursorVec branches_;
};

CursorPtr merge_choices(CursorVec choices) {
  CursorVec flattened;
  for (auto& choice : choices) {
    if (choice) {
      flattened.push_back(std::move(choice));
    }
  }
  if (flattened.empty()) {
    return nullptr;
  }
  if (flattened.size() == 1) {
    return flattened.front();
  }
  return std::make_shared<AnyOfCursor>(std::move(flattened));
}

CursorPtr intersect_choices(CursorVec branches) {
  CursorVec flattened;
  for (auto& branch : branches) {
    if (branch) {
      flattened.push_back(std::move(branch));
    }
  }
  if (flattened.empty()) {
    return nullptr;
  }
  if (flattened.size() == 1) {
    return flattened.front();
  }
  return std::make_shared<AllOfCursor>(std::move(flattened));
}

class LiteralCursor final : public JsonSchemaCursor {
 public:
  explicit LiteralCursor(std::string literal, size_t offset = 0)
      : literal_(std::move(literal)), offset_(offset) {}

  void advance(unsigned char c, CursorVec& out) const override {
    if (offset_ < literal_.size() &&
        c == static_cast<unsigned char>(literal_[offset_])) {
      out.push_back(std::make_shared<LiteralCursor>(literal_, offset_ + 1));
    }
  }

  bool is_complete() const override { return offset_ == literal_.size(); }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    if (offset_ < literal_.size()) {
      add_byte(mask, static_cast<unsigned char>(literal_[offset_]));
    }
  }

 private:
  std::string literal_;
  size_t offset_;
};

class StringCursor final : public JsonSchemaCursor {
 public:
  enum class Phase { START, BODY, ESCAPE, UNICODE, DONE };

  explicit StringCursor(Phase phase = Phase::START, int unicode_left = 0)
      : phase_(phase), unicode_left_(unicode_left) {}

  void advance(unsigned char c, CursorVec& out) const override {
    switch (phase_) {
      case Phase::START:
        if (c == '"') {
          out.push_back(std::make_shared<StringCursor>(Phase::BODY, 0));
        }
        return;
      case Phase::BODY:
        if (c == '"') {
          out.push_back(std::make_shared<StringCursor>(Phase::DONE, 0));
          return;
        }
        if (c == '\\') {
          out.push_back(std::make_shared<StringCursor>(Phase::ESCAPE, 0));
          return;
        }
        if (c >= 0x20) {
          out.push_back(std::make_shared<StringCursor>(Phase::BODY, 0));
        }
        return;
      case Phase::ESCAPE:
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
            c == 'n' || c == 'r' || c == 't') {
          out.push_back(std::make_shared<StringCursor>(Phase::BODY, 0));
          return;
        }
        if (c == 'u') {
          out.push_back(std::make_shared<StringCursor>(Phase::UNICODE, 4));
        }
        return;
      case Phase::UNICODE:
        if (std::isxdigit(c) != 0) {
          const int next_left = unicode_left_ - 1;
          out.push_back(std::make_shared<StringCursor>(
              next_left == 0 ? Phase::BODY : Phase::UNICODE, next_left));
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_complete() const override { return phase_ == Phase::DONE; }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    switch (phase_) {
      case Phase::START:
        add_byte(mask, '"');
        return;
      case Phase::BODY:
        add_byte(mask, '"');
        add_byte(mask, '\\');
        for (int i = 0x20; i < 256; ++i) {
          if (i == '"' || i == '\\') {
            continue;
          }
          add_byte(mask, static_cast<unsigned char>(i));
        }
        return;
      case Phase::ESCAPE:
        for (unsigned char c : std::string("\"\\/bfnrtu")) {
          add_byte(mask, c);
        }
        return;
      case Phase::UNICODE:
        for (unsigned char c : std::string("0123456789abcdefABCDEF")) {
          add_byte(mask, c);
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_open_ended_text_frontier() const override {
    return phase_ == Phase::BODY || phase_ == Phase::ESCAPE ||
           phase_ == Phase::UNICODE;
  }

 private:
  Phase phase_;
  int unicode_left_;
};

class NumberCursor final : public JsonSchemaCursor {
 public:
  enum class Phase {
    START,
    SIGN,
    ZERO,
    INT,
    DOT,
    FRAC,
    EXP_MARK,
    EXP_SIGN,
    EXP,
  };

  explicit NumberCursor(Phase phase = Phase::START) : phase_(phase) {}

  void advance(unsigned char c, CursorVec& out) const override {
    switch (phase_) {
      case Phase::START:
        if (c == '-') {
          out.push_back(std::make_shared<NumberCursor>(Phase::SIGN));
        } else if (c == '0') {
          out.push_back(std::make_shared<NumberCursor>(Phase::ZERO));
        } else if (c >= '1' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::INT));
        }
        return;
      case Phase::SIGN:
        if (c == '0') {
          out.push_back(std::make_shared<NumberCursor>(Phase::ZERO));
        } else if (c >= '1' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::INT));
        }
        return;
      case Phase::ZERO:
        if (c == '.') {
          out.push_back(std::make_shared<NumberCursor>(Phase::DOT));
        } else if (c == 'e' || c == 'E') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP_MARK));
        }
        return;
      case Phase::INT:
        if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::INT));
        } else if (c == '.') {
          out.push_back(std::make_shared<NumberCursor>(Phase::DOT));
        } else if (c == 'e' || c == 'E') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP_MARK));
        }
        return;
      case Phase::DOT:
        if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::FRAC));
        }
        return;
      case Phase::FRAC:
        if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::FRAC));
        } else if (c == 'e' || c == 'E') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP_MARK));
        }
        return;
      case Phase::EXP_MARK:
        if (c == '+' || c == '-') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP_SIGN));
        } else if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP));
        }
        return;
      case Phase::EXP_SIGN:
        if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP));
        }
        return;
      case Phase::EXP:
        if (c >= '0' && c <= '9') {
          out.push_back(std::make_shared<NumberCursor>(Phase::EXP));
        }
        return;
    }
  }

  bool is_complete() const override {
    return phase_ == Phase::ZERO || phase_ == Phase::INT ||
           phase_ == Phase::FRAC || phase_ == Phase::EXP;
  }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    switch (phase_) {
      case Phase::START:
        add_byte(mask, '-');
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        return;
      case Phase::SIGN:
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        return;
      case Phase::ZERO:
        add_byte(mask, '.');
        add_byte(mask, 'e');
        add_byte(mask, 'E');
        return;
      case Phase::INT:
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        add_byte(mask, '.');
        add_byte(mask, 'e');
        add_byte(mask, 'E');
        return;
      case Phase::DOT:
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        return;
      case Phase::FRAC:
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        add_byte(mask, 'e');
        add_byte(mask, 'E');
        return;
      case Phase::EXP_MARK:
        add_byte(mask, '+');
        add_byte(mask, '-');
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        return;
      case Phase::EXP_SIGN:
      case Phase::EXP:
        for (unsigned char c = '0'; c <= '9'; ++c) {
          add_byte(mask, c);
        }
        return;
    }
  }

 private:
  Phase phase_;
};

bool has_schema_keyword(const nlohmann::json& schema) {
  if (!schema.is_object()) {
    return false;
  }
  for (const auto& key : {"type",
                          "properties",
                          "required",
                          "items",
                          "enum",
                          "const",
                          "anyOf",
                          "oneOf",
                          "allOf",
                          "$ref"}) {
    if (schema.contains(key)) {
      return true;
    }
  }
  return false;
}

nlohmann::json resolve_local_ref(const nlohmann::json& root,
                                 const std::string& ref) {
  if (ref.empty() || ref[0] != '#') {
    return nullptr;
  }
  if (ref == "#") {
    return root;
  }
  try {
    return root.at(nlohmann::json::json_pointer(ref.substr(1)));
  } catch (const std::exception&) {
    return nullptr;
  }
}

nlohmann::json normalize_json_schema(const nlohmann::json& schema,
                                     const nlohmann::json& root,
                                     int depth = 0) {
  if (depth > 64) {
    return schema;
  }
  if (schema.is_array()) {
    nlohmann::json normalized = nlohmann::json::array();
    for (const auto& item : schema) {
      normalized.push_back(normalize_json_schema(item, root, depth + 1));
    }
    return normalized;
  }
  if (!schema.is_object()) {
    return schema;
  }

  if (schema.contains("$ref") && schema["$ref"].is_string()) {
    auto resolved = resolve_local_ref(root, schema["$ref"].get<std::string>());
    if (!resolved.is_null()) {
      auto merged = normalize_json_schema(resolved, root, depth + 1);
      for (const auto& [key, value] : schema.items()) {
        if (key == "$ref") {
          continue;
        }
        if (key == "properties" && merged[key].is_object() &&
            value.is_object()) {
          for (const auto& [prop_key, prop_value] : value.items()) {
            merged[key][prop_key] =
                normalize_json_schema(prop_value, root, depth + 1);
          }
          continue;
        }
        if (key == "required" && merged[key].is_array() && value.is_array()) {
          std::set<std::string> merged_required;
          for (const auto& item : merged[key]) {
            if (item.is_string()) {
              merged_required.insert(item.get<std::string>());
            }
          }
          for (const auto& item : value) {
            if (item.is_string()) {
              merged_required.insert(item.get<std::string>());
            }
          }
          merged[key] = nlohmann::json::array();
          for (const auto& item : merged_required) {
            merged[key].push_back(item);
          }
          continue;
        }
        merged[key] = normalize_json_schema(value, root, depth + 1);
      }
      return merged;
    }
  }

  nlohmann::json normalized = schema;
  for (auto& [key, value] : normalized.items()) {
    if (key == "properties" && value.is_object()) {
      for (auto& [prop_key, prop_value] : value.items()) {
        prop_value = normalize_json_schema(prop_value, root, depth + 1);
      }
      continue;
    }
    if ((key == "$defs" || key == "defs") && value.is_object()) {
      for (auto& [def_key, def_value] : value.items()) {
        def_value = normalize_json_schema(def_value, root, depth + 1);
      }
      continue;
    }
    if ((key == "anyOf" || key == "oneOf" || key == "allOf" ||
         key == "prefixItems") &&
        value.is_array()) {
      for (auto& item : value) {
        item = normalize_json_schema(item, root, depth + 1);
      }
      continue;
    }
    if (key == "items") {
      value = normalize_json_schema(value, root, depth + 1);
    }
  }
  return normalized;
}

CursorPtr create_cursor_from_schema(const nlohmann::json& schema);

class ArrayCursor final : public JsonSchemaCursor {
 public:
  enum class Phase { START, AFTER_OPEN, AFTER_COMMA, ITEM, AFTER_ITEM, DONE };

  ArrayCursor(nlohmann::json item_schema,
              size_t min_items,
              int64_t max_items,
              Phase phase = Phase::START,
              size_t count = 0,
              CursorPtr child = nullptr)
      : item_schema_(std::move(item_schema)),
        min_items_(min_items),
        max_items_(max_items),
        phase_(phase),
        count_(count),
        child_(std::move(child)) {}

  void advance(unsigned char c, CursorVec& out) const override {
    switch (phase_) {
      case Phase::START:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ArrayCursor>(*this));
        } else if (c == '[') {
          out.push_back(std::make_shared<ArrayCursor>(
              item_schema_, min_items_, max_items_, Phase::AFTER_OPEN, count_));
        }
        return;
      case Phase::AFTER_OPEN:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ArrayCursor>(*this));
          return;
        }
        if (count_ >= min_items_ && c == ']') {
          out.push_back(std::make_shared<ArrayCursor>(
              item_schema_, min_items_, max_items_, Phase::DONE, count_));
          return;
        }
        if (max_items_ >= 0 && static_cast<int64_t>(count_) >= max_items_) {
          return;
        }
        if (auto item = create_cursor_from_schema(item_schema_)) {
          CursorVec item_out;
          item->advance(c, item_out);
          if (!item_out.empty()) {
            out.push_back(std::make_shared<ArrayCursor>(
                item_schema_,
                min_items_,
                max_items_,
                Phase::ITEM,
                count_,
                merge_choices(std::move(item_out))));
          }
        }
        return;
      case Phase::AFTER_COMMA:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ArrayCursor>(*this));
          return;
        }
        if (max_items_ >= 0 && static_cast<int64_t>(count_) >= max_items_) {
          return;
        }
        if (auto item = create_cursor_from_schema(item_schema_)) {
          CursorVec item_out;
          item->advance(c, item_out);
          if (!item_out.empty()) {
            out.push_back(std::make_shared<ArrayCursor>(
                item_schema_,
                min_items_,
                max_items_,
                Phase::ITEM,
                count_,
                merge_choices(std::move(item_out))));
          }
        }
        return;
      case Phase::ITEM: {
        CursorVec child_out;
        child_->advance(c, child_out);
        if (!child_out.empty()) {
          out.push_back(std::make_shared<ArrayCursor>(
              item_schema_,
              min_items_,
              max_items_,
              Phase::ITEM,
              count_,
              merge_choices(std::move(child_out))));
        }
        if (!child_->is_complete()) {
          return;
        }
        const size_t next_count = count_ + 1;
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ArrayCursor>(item_schema_,
                                                      min_items_,
                                                      max_items_,
                                                      Phase::AFTER_ITEM,
                                                      next_count));
        } else if (c == ',' &&
                   !(max_items_ >= 0 &&
                     static_cast<int64_t>(next_count) >= max_items_)) {
          out.push_back(std::make_shared<ArrayCursor>(item_schema_,
                                                      min_items_,
                                                      max_items_,
                                                      Phase::AFTER_COMMA,
                                                      next_count));
        } else if (c == ']' && next_count >= min_items_) {
          out.push_back(std::make_shared<ArrayCursor>(
              item_schema_, min_items_, max_items_, Phase::DONE, next_count));
        }
        return;
      }
      case Phase::AFTER_ITEM:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ArrayCursor>(*this));
        } else if (c == ',' && !(max_items_ >= 0 &&
                                 static_cast<int64_t>(count_) >= max_items_)) {
          out.push_back(std::make_shared<ArrayCursor>(item_schema_,
                                                      min_items_,
                                                      max_items_,
                                                      Phase::AFTER_COMMA,
                                                      count_));
        } else if (c == ']' && count_ >= min_items_) {
          out.push_back(std::make_shared<ArrayCursor>(
              item_schema_, min_items_, max_items_, Phase::DONE, count_));
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_complete() const override { return phase_ == Phase::DONE; }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    switch (phase_) {
      case Phase::START:
        add_whitespace(mask);
        add_byte(mask, '[');
        return;
      case Phase::AFTER_OPEN:
        add_whitespace(mask);
        if (count_ >= min_items_) {
          add_byte(mask, ']');
        }
        if (!(max_items_ >= 0 && static_cast<int64_t>(count_) >= max_items_)) {
          if (auto item = create_cursor_from_schema(item_schema_)) {
            item->collect_next_bytes(mask);
          }
        }
        return;
      case Phase::AFTER_COMMA:
        add_whitespace(mask);
        if (!(max_items_ >= 0 && static_cast<int64_t>(count_) >= max_items_)) {
          if (auto item = create_cursor_from_schema(item_schema_)) {
            item->collect_next_bytes(mask);
          }
        }
        return;
      case Phase::ITEM:
        child_->collect_next_bytes(mask);
        if (child_->is_complete()) {
          add_whitespace(mask);
          if (!(max_items_ >= 0 &&
                static_cast<int64_t>(count_ + 1) >= max_items_)) {
            add_byte(mask, ',');
          }
          if (count_ + 1 >= min_items_) {
            add_byte(mask, ']');
          }
        }
        return;
      case Phase::AFTER_ITEM:
        add_whitespace(mask);
        if (!(max_items_ >= 0 && static_cast<int64_t>(count_) >= max_items_)) {
          add_byte(mask, ',');
        }
        if (count_ >= min_items_) {
          add_byte(mask, ']');
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_open_ended_text_frontier() const override {
    return phase_ == Phase::ITEM && child_ != nullptr &&
           child_->is_open_ended_text_frontier();
  }

 private:
  nlohmann::json item_schema_;
  size_t min_items_;
  int64_t max_items_;
  Phase phase_;
  size_t count_;
  CursorPtr child_;
};

class ObjectCursor final : public JsonSchemaCursor {
 public:
  enum class Phase {
    START,
    AFTER_OPEN,
    AFTER_COMMA,
    KEY,
    AFTER_KEY,
    BEFORE_VALUE,
    VALUE,
    AFTER_VALUE,
    DONE,
  };

  ObjectCursor(nlohmann::json properties,
               std::set<std::string> required_remaining,
               Phase phase = Phase::START,
               std::set<std::string> seen_keys = {},
               std::set<std::string> key_candidates = {},
               std::string key_prefix = "",
               std::string selected_key = "",
               CursorPtr child = nullptr)
      : properties_(std::move(properties)),
        required_remaining_(std::move(required_remaining)),
        phase_(phase),
        seen_keys_(std::move(seen_keys)),
        key_candidates_(std::move(key_candidates)),
        key_prefix_(std::move(key_prefix)),
        selected_key_(std::move(selected_key)),
        child_(std::move(child)) {}

  bool has_available_keys(const std::set<std::string>& seen_keys) const {
    for (const auto& [key, _] : properties_.items()) {
      if (seen_keys.count(key) == 0) {
        return true;
      }
    }
    return false;
  }

  void advance(unsigned char c, CursorVec& out) const override {
    switch (phase_) {
      case Phase::START:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
        } else if (c == '{') {
          out.push_back(std::make_shared<ObjectCursor>(
              properties_, required_remaining_, Phase::AFTER_OPEN, seen_keys_));
        }
        return;
      case Phase::AFTER_OPEN:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
          return;
        }
        if (required_remaining_.empty() && c == '}') {
          out.push_back(std::make_shared<ObjectCursor>(
              properties_, required_remaining_, Phase::DONE, seen_keys_));
          return;
        }
        [[fallthrough]];
      case Phase::AFTER_COMMA:
        if (phase_ == Phase::AFTER_COMMA && is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
          return;
        }
        if (c == '"') {
          std::set<std::string> candidates;
          const std::string prefix(1, static_cast<char>(c));
          for (const auto& [key, _] : properties_.items()) {
            if (seen_keys_.count(key) > 0) {
              continue;
            }
            const auto literal = json_dump(key);
            if (literal.rfind(prefix, 0) == 0) {
              candidates.insert(key);
            }
          }
          if (!candidates.empty()) {
            out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                         required_remaining_,
                                                         Phase::KEY,
                                                         seen_keys_,
                                                         std::move(candidates),
                                                         prefix));
          }
        }
        return;
      case Phase::KEY: {
        std::set<std::string> next_candidates;
        const std::string next_prefix = key_prefix_ + static_cast<char>(c);
        std::string selected;
        for (const auto& key : key_candidates_) {
          const auto literal = json_dump(key);
          if (literal.rfind(next_prefix, 0) == 0) {
            next_candidates.insert(key);
            if (literal == next_prefix) {
              selected = key;
            }
          }
        }
        if (next_candidates.empty()) {
          return;
        }
        out.push_back(std::make_shared<ObjectCursor>(
            properties_,
            required_remaining_,
            selected.empty() ? Phase::KEY : Phase::AFTER_KEY,
            seen_keys_,
            std::move(next_candidates),
            next_prefix,
            selected));
        return;
      }
      case Phase::AFTER_KEY:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
        } else if (c == ':') {
          out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                       required_remaining_,
                                                       Phase::BEFORE_VALUE,
                                                       seen_keys_,
                                                       std::set<std::string>{},
                                                       "",
                                                       selected_key_));
        }
        return;
      case Phase::BEFORE_VALUE:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
          return;
        }
        if (!properties_.contains(selected_key_)) {
          return;
        }
        if (auto value =
                create_cursor_from_schema(properties_[selected_key_])) {
          CursorVec value_out;
          value->advance(c, value_out);
          if (!value_out.empty()) {
            out.push_back(std::make_shared<ObjectCursor>(
                properties_,
                required_remaining_,
                Phase::VALUE,
                seen_keys_,
                std::set<std::string>{},
                "",
                selected_key_,
                merge_choices(std::move(value_out))));
          }
        }
        return;
      case Phase::VALUE: {
        CursorVec child_out;
        child_->advance(c, child_out);
        if (!child_out.empty()) {
          out.push_back(std::make_shared<ObjectCursor>(
              properties_,
              required_remaining_,
              Phase::VALUE,
              seen_keys_,
              std::set<std::string>{},
              "",
              selected_key_,
              merge_choices(std::move(child_out))));
        }
        if (!child_->is_complete()) {
          return;
        }
        auto next_seen = seen_keys_;
        next_seen.insert(selected_key_);
        auto next_required = required_remaining_;
        next_required.erase(selected_key_);
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                       std::move(next_required),
                                                       Phase::AFTER_VALUE,
                                                       std::move(next_seen)));
        } else if (c == ',' && has_available_keys(next_seen)) {
          out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                       std::move(next_required),
                                                       Phase::AFTER_COMMA,
                                                       std::move(next_seen)));
        } else if (c == '}' && next_required.empty()) {
          out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                       std::move(next_required),
                                                       Phase::DONE,
                                                       std::move(next_seen)));
        }
        return;
      }
      case Phase::AFTER_VALUE:
        if (is_json_whitespace(c)) {
          out.push_back(std::make_shared<ObjectCursor>(*this));
        } else if (c == ',' && has_available_keys(seen_keys_)) {
          out.push_back(std::make_shared<ObjectCursor>(properties_,
                                                       required_remaining_,
                                                       Phase::AFTER_COMMA,
                                                       seen_keys_));
        } else if (c == '}' && required_remaining_.empty()) {
          out.push_back(std::make_shared<ObjectCursor>(
              properties_, required_remaining_, Phase::DONE, seen_keys_));
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_complete() const override { return phase_ == Phase::DONE; }

  void collect_next_bytes(std::array<bool, 256>& mask) const override {
    switch (phase_) {
      case Phase::START:
        add_whitespace(mask);
        add_byte(mask, '{');
        return;
      case Phase::AFTER_OPEN:
        add_whitespace(mask);
        if (required_remaining_.empty()) {
          add_byte(mask, '}');
        }
        if (has_available_keys(seen_keys_)) {
          add_byte(mask, '"');
        }
        return;
      case Phase::AFTER_COMMA:
        if (has_available_keys(seen_keys_)) {
          add_whitespace(mask);
          add_byte(mask, '"');
        }
        return;
      case Phase::KEY:
        for (const auto& key : key_candidates_) {
          const auto literal = json_dump(key);
          if (literal.size() > key_prefix_.size()) {
            add_byte(mask,
                     static_cast<unsigned char>(literal[key_prefix_.size()]));
          }
        }
        return;
      case Phase::AFTER_KEY:
        add_whitespace(mask);
        add_byte(mask, ':');
        return;
      case Phase::BEFORE_VALUE:
        add_whitespace(mask);
        if (properties_.contains(selected_key_)) {
          if (auto value =
                  create_cursor_from_schema(properties_[selected_key_])) {
            value->collect_next_bytes(mask);
          }
        }
        return;
      case Phase::VALUE:
        child_->collect_next_bytes(mask);
        if (child_->is_complete()) {
          add_whitespace(mask);
          auto next_required = required_remaining_;
          next_required.erase(selected_key_);
          auto next_seen = seen_keys_;
          next_seen.insert(selected_key_);
          if (has_available_keys(next_seen)) {
            add_byte(mask, ',');
          }
          if (next_required.empty()) {
            add_byte(mask, '}');
          }
        }
        return;
      case Phase::AFTER_VALUE:
        add_whitespace(mask);
        if (has_available_keys(seen_keys_)) {
          add_byte(mask, ',');
        }
        if (required_remaining_.empty()) {
          add_byte(mask, '}');
        }
        return;
      case Phase::DONE:
        return;
    }
  }

  bool is_open_ended_text_frontier() const override {
    return phase_ == Phase::VALUE && child_ != nullptr &&
           child_->is_open_ended_text_frontier();
  }

 private:
  nlohmann::json properties_;
  std::set<std::string> required_remaining_;
  Phase phase_;
  std::set<std::string> seen_keys_;
  std::set<std::string> key_candidates_;
  std::string key_prefix_;
  std::string selected_key_;
  CursorPtr child_;
};

CursorPtr create_cursor_from_schema(const nlohmann::json& schema) {
  if (schema.is_object()) {
    if (schema.contains("enum") && schema["enum"].is_array()) {
      CursorVec literals;
      for (const auto& value : schema["enum"]) {
        literals.push_back(std::make_shared<LiteralCursor>(json_dump(value)));
      }
      return merge_choices(std::move(literals));
    }

    if (schema.contains("const")) {
      return std::make_shared<LiteralCursor>(json_dump(schema["const"]));
    }

    for (const auto& key : {"anyOf", "oneOf"}) {
      if (schema.contains(key) && schema[key].is_array()) {
        CursorVec branches;
        for (const auto& branch : schema[key]) {
          if (auto cursor = create_cursor_from_schema(branch)) {
            branches.push_back(cursor);
          }
        }
        return merge_choices(std::move(branches));
      }
    }

    if (schema.contains("allOf") && schema["allOf"].is_array()) {
      CursorVec branches;
      for (const auto& branch : schema["allOf"]) {
        if (auto cursor = create_cursor_from_schema(branch)) {
          branches.push_back(cursor);
        }
      }
      auto remaining = schema;
      remaining.erase("allOf");
      if (has_schema_keyword(remaining)) {
        if (auto cursor = create_cursor_from_schema(remaining)) {
          branches.push_back(cursor);
        }
      }
      return intersect_choices(std::move(branches));
    }

    if (schema.contains("type") && schema["type"].is_array()) {
      CursorVec branches;
      for (const auto& type_item : schema["type"]) {
        if (!type_item.is_string()) {
          continue;
        }
        auto branch = schema;
        branch["type"] = type_item.get<std::string>();
        if (auto cursor = create_cursor_from_schema(branch)) {
          branches.push_back(cursor);
        }
      }
      return merge_choices(std::move(branches));
    }

    if (schema_allows_type(schema, "object") || schema.contains("properties")) {
      std::set<std::string> required;
      if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto& item : schema["required"]) {
          if (item.is_string()) {
            required.insert(item.get<std::string>());
          }
        }
      }
      return std::make_shared<ObjectCursor>(
          schema.value("properties", nlohmann::json::object()),
          std::move(required));
    }

    if (schema_allows_type(schema, "array") || schema.contains("items")) {
      size_t min_items = schema.value("minItems", 0);
      int64_t max_items =
          schema.contains("maxItems") && schema["maxItems"].is_number_integer()
              ? schema["maxItems"].get<int64_t>()
              : -1;
      return std::make_shared<ArrayCursor>(
          schema.value("items", nlohmann::json::object()),
          min_items,
          max_items);
    }

    if (schema_allows_type(schema, "string")) {
      return std::make_shared<StringCursor>();
    }

    if (schema_allows_type(schema, "integer") ||
        schema_allows_type(schema, "number")) {
      return std::make_shared<NumberCursor>();
    }

    if (schema_allows_type(schema, "boolean")) {
      return merge_choices({std::make_shared<LiteralCursor>("true"),
                            std::make_shared<LiteralCursor>("false")});
    }

    if (schema_allows_type(schema, "null")) {
      return std::make_shared<LiteralCursor>("null");
    }
  }

  return merge_choices(
      {std::make_shared<ObjectCursor>(nlohmann::json::object(),
                                      std::set<std::string>{}),
       std::make_shared<ArrayCursor>(nlohmann::json::object(), 0, -1),
       std::make_shared<StringCursor>(),
       std::make_shared<NumberCursor>(),
       std::make_shared<LiteralCursor>("true"),
       std::make_shared<LiteralCursor>("false"),
       std::make_shared<LiteralCursor>("null")});
}

CursorVec advance_all(const CursorVec& states, std::string_view text) {
  CursorVec active = states;
  for (unsigned char c : text) {
    CursorVec next;
    for (const auto& state : active) {
      state->advance(c, next);
    }
    active = std::move(next);
    if (active.empty()) {
      break;
    }
  }
  return active;
}

std::array<bool, 256> collect_next_bytes(const CursorVec& states) {
  std::array<bool, 256> mask{};
  mask.fill(false);
  for (const auto& state : states) {
    state->collect_next_bytes(mask);
  }
  return mask;
}

nlohmann::json collect_tool_schema_defs(
    const std::vector<function_call::JsonTool>& tools) {
  nlohmann::json defs = nlohmann::json::object();
  for (const auto& tool : tools) {
    if (!tool.function.parameters.is_object()) {
      continue;
    }
    for (const auto& defs_key : {"$defs", "defs"}) {
      if (!tool.function.parameters.contains(defs_key) ||
          !tool.function.parameters[defs_key].is_object()) {
        continue;
      }
      for (const auto& [def_name, def_schema] :
           tool.function.parameters[defs_key].items()) {
        if (defs.contains(def_name) && defs[def_name] != def_schema) {
          LOG(WARNING) << "Conflicting tool schema definition for $" << defs_key
                       << "." << def_name << ", later definition is ignored";
          continue;
        }
        defs[def_name] = def_schema;
      }
    }
  }
  return defs;
}

nlohmann::json build_tool_schema_for_choice(
    const std::vector<function_call::JsonTool>& tools,
    ToolCallConstraintMode mode) {
  nlohmann::json branches = nlohmann::json::array();
  for (const auto& tool : tools) {
    branches.push_back(nlohmann::json{
        {"type", "object"},
        {"properties",
         {{"name", {{"enum", nlohmann::json::array({tool.function.name})}}},
          {"parameters",
           tool.function.parameters.is_object() &&
                   !tool.function.parameters.empty()
               ? tool.function.parameters
               : nlohmann::json{{"type", "object"}}}}},
        {"required", nlohmann::json::array({"name", "parameters"})}});
  }

  nlohmann::json schema = {
      {"type", "array"},
      {"minItems", 1},
      {"items", {{"type", "object"}, {"anyOf", branches}}}};
  auto defs = collect_tool_schema_defs(tools);
  if (!defs.empty()) {
    schema["$defs"] = std::move(defs);
  }
  if (mode == ToolCallConstraintMode::NAMED) {
    schema["maxItems"] = 1;
  }
  return schema;
}

}  // namespace

ToolCallConstrainedDecoding::ToolCallConstrainedDecoding(
    const Tokenizer& tokenizer,
    int32_t vocab_size,
    torch::ScalarType dtype,
    torch::Device device,
    const std::vector<ToolCallConstraintMode>& modes,
    const std::vector<std::vector<std::string>>& allowed_tool_names_vec,
    const std::vector<std::vector<std::string>>& allowed_tool_schema_jsons_vec)
    : tokenizer_(tokenizer),
      vocab_size_(vocab_size),
      dtype_(dtype),
      device_(device),
      modes_(modes),
      allowed_tool_names_vec_(allowed_tool_names_vec),
      allowed_tool_schema_jsons_vec_(allowed_tool_schema_jsons_vec) {}

bool ToolCallConstrainedDecoding::build_mask_cache() {
  build_token_cache();
  compiled_grammars_.clear();
  compiled_grammars_.reserve(allowed_tool_names_vec_.size());
  root_cursors_.clear();
  root_cursors_.reserve(allowed_tool_names_vec_.size());
  for (size_t i = 0; i < allowed_tool_names_vec_.size(); ++i) {
    auto compiled = build_compiled_grammar_for_sequence(i);
    compiled_grammars_.push_back(compiled);
    root_cursors_.push_back(compiled ? nullptr
                                     : build_root_cursor_for_sequence(i));
  }
  return true;
}

void ToolCallConstrainedDecoding::build_token_cache() {
  if (token_cache_) {
    return;
  }
  token_cache_ = get_or_build_token_cache(tokenizer_, vocab_size_);
}

std::vector<function_call::JsonTool>
ToolCallConstrainedDecoding::parse_tools_for_sequence(size_t index) const {
  std::vector<function_call::JsonTool> tools;
  if (index >= allowed_tool_schema_jsons_vec_.size()) {
    return tools;
  }

  tools.reserve(allowed_tool_schema_jsons_vec_[index].size());
  for (const auto& tool_json : allowed_tool_schema_jsons_vec_[index]) {
    if (tool_json.empty()) {
      continue;
    }
    try {
      auto obj = nlohmann::json::parse(tool_json);
      function_call::JsonTool tool;
      tool.type = obj.value("type", "function");
      if (obj.contains("function") && obj["function"].is_object()) {
        const auto& function = obj["function"];
        tool.function.name = function.value("name", "");
        tool.function.description = function.value("description", "");
        if (function.contains("parameters")) {
          tool.function.parameters = function["parameters"];
        }
      }
      if (!tool.function.name.empty()) {
        tools.push_back(std::move(tool));
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to parse tool schema JSON: " << e.what();
    }
  }
  return tools;
}

std::shared_ptr<const xgrammar::CompiledGrammar>
ToolCallConstrainedDecoding::build_compiled_grammar_for_sequence(
    size_t index) const {
  if (index >= modes_.size() || modes_[index] == ToolCallConstraintMode::NONE ||
      !token_cache_ || !token_cache_->xgrammar_compiler) {
    return nullptr;
  }

  auto tools = parse_tools_for_sequence(index);
  if (tools.empty()) {
    return nullptr;
  }

  const std::string schema_json =
      json_dump(build_tool_schema_for_choice(tools, modes_[index]));

  static std::mutex cache_mutex;
  static std::unordered_map<std::string,
                            std::weak_ptr<const xgrammar::CompiledGrammar>>
      cache_by_key;

  std::string cache_key = std::to_string(
      reinterpret_cast<uintptr_t>(token_cache_->xgrammar_compiler.get()));
  cache_key.push_back('\n');
  cache_key.append(schema_json);

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache_by_key.find(cache_key);
    if (it != cache_by_key.end()) {
      if (auto cached = it->second.lock()) {
        return cached;
      }
    }
  }

  std::shared_ptr<const xgrammar::CompiledGrammar> compiled;
  try {
    compiled = std::make_shared<xgrammar::CompiledGrammar>(
        token_cache_->xgrammar_compiler->CompileJSONSchema(schema_json));
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to compile tool schema with xgrammar: " << e.what();
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& slot = cache_by_key[cache_key];
  if (auto cached = slot.lock()) {
    return cached;
  }
  slot = compiled;
  return compiled;
}

std::shared_ptr<const JsonSchemaCursor>
ToolCallConstrainedDecoding::build_root_cursor_for_sequence(
    size_t index) const {
  if (index >= modes_.size() || modes_[index] == ToolCallConstraintMode::NONE) {
    return nullptr;
  }
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::weak_ptr<const JsonSchemaCursor>>
      cache_by_key;

  std::string cache_key = std::to_string(static_cast<int32_t>(modes_[index]));
  cache_key.push_back('\n');
  if (index < allowed_tool_schema_jsons_vec_.size()) {
    for (const auto& tool_json : allowed_tool_schema_jsons_vec_[index]) {
      cache_key.append(tool_json);
      cache_key.push_back('\x1f');
    }
  }

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache_by_key.find(cache_key);
    if (it != cache_by_key.end()) {
      if (auto cached = it->second.lock()) {
        return cached;
      }
    }
  }

  auto tools = parse_tools_for_sequence(index);
  if (tools.empty()) {
    return nullptr;
  }
  auto schema = build_tool_schema_for_choice(tools, modes_[index]);
  schema = normalize_json_schema(schema, schema);
  auto cursor = create_cursor_from_schema(schema);
  if (!cursor) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto& slot = cache_by_key[cache_key];
  if (auto cached = slot.lock()) {
    return cached;
  }
  slot = cursor;
  return cursor;
}

torch::Tensor ToolCallConstrainedDecoding::generate_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  if (generated_token_list.empty() ||
      (compiled_grammars_.empty() && root_cursors_.empty())) {
    return torch::Tensor();
  }

  auto options = torch::TensorOptions().dtype(dtype_).device(device_);
  auto mask = torch::zeros(
      {static_cast<int64_t>(generated_token_list.size()), vocab_size_},
      options);

  bool any_constrained = false;
  auto apply_cursor_constraint =
      [&](size_t seq_idx, const std::vector<int32_t>& generated_ids) {
        if (seq_idx >= root_cursors_.size() || !root_cursors_[seq_idx]) {
          return;
        }

        std::string generated_text;
        if (!generated_ids.empty()) {
          generated_text = tokenizer_.decode(
              Slice<int32_t>(generated_ids.data(), generated_ids.size()),
              /*skip_special_tokens=*/false);
        }

        CursorVec active = {root_cursors_[seq_idx]};
        active = advance_all(active, generated_text);
        if (active.empty()) {
          return;
        }

        if (std::any_of(
                active.begin(), active.end(), [](const CursorPtr& state) {
                  return state->is_complete();
                })) {
          return;
        }

        if (std::all_of(
                active.begin(), active.end(), [](const CursorPtr& state) {
                  return state->is_open_ended_text_frontier();
                })) {
          return;
        }

        const auto next_bytes = collect_next_bytes(active);
        std::unordered_set<int32_t> allowed_token_ids;
        for (size_t byte = 0; byte < next_bytes.size(); ++byte) {
          if (!next_bytes[byte]) {
            continue;
          }
          for (int32_t token_id : token_cache_->token_ids_by_first_byte[byte]) {
            const auto& token_text = token_cache_->token_text_cache[token_id];
            if (token_text.empty()) {
              continue;
            }
            if (!advance_all(active, token_text).empty()) {
              allowed_token_ids.insert(token_id);
            }
          }
        }

        if (allowed_token_ids.empty()) {
          return;
        }

        any_constrained = true;
        auto row =
            torch::full({vocab_size_}, PRE_MASK_FACTOR, torch::dtype(dtype_));
        for (int32_t token_id : allowed_token_ids) {
          if (token_id >= 0 && token_id < vocab_size_) {
            row[token_id] = 0.0f;
          }
        }
        mask.index_put_({static_cast<int64_t>(seq_idx)},
                        safe_to(row, device_, true));
      };

  for (size_t i = 0; i < generated_token_list.size(); ++i) {
    const auto& generated_ids = generated_token_list[i];
    bool handled_by_xgrammar = false;
    if (i < compiled_grammars_.size() && compiled_grammars_[i]) {
      try {
        xgrammar::GrammarMatcher matcher(*compiled_grammars_[i]);
        bool accepted = true;
        for (int32_t token_id : generated_ids) {
          if (!matcher.AcceptToken(token_id)) {
            accepted = false;
            break;
          }
        }

        if (accepted) {
          handled_by_xgrammar = true;
          if (!matcher.IsCompleted()) {
            std::vector<int32_t> bitmask_buffer(
                xgrammar::GetBitmaskSize(vocab_size_));
            int64_t shape = static_cast<int64_t>(bitmask_buffer.size());
            int64_t stride = 1;
            DLTensor bitmask_tensor{bitmask_buffer.data(),
                                    DLDevice{kDLCPU, 0},
                                    1,
                                    xgrammar::GetBitmaskDLType(),
                                    &shape,
                                    &stride,
                                    0};
            const bool need_apply =
                matcher.FillNextTokenBitmask(&bitmask_tensor);
            if (need_apply) {
              std::vector<int> rejected_token_ids;
              xgrammar::_DebugGetMaskedTokensFromBitmask(
                  &rejected_token_ids, bitmask_tensor, vocab_size_);
              if (!rejected_token_ids.empty()) {
                any_constrained = true;
                auto row = torch::zeros({vocab_size_}, torch::dtype(dtype_));
                for (int32_t token_id : rejected_token_ids) {
                  if (token_id >= 0 && token_id < vocab_size_) {
                    row[token_id] = PRE_MASK_FACTOR;
                  }
                }
                mask.index_put_({static_cast<int64_t>(i)},
                                safe_to(row, device_, true));
              }
            }
          }
        }
      } catch (const std::exception& e) {
        LOG(WARNING)
            << "xgrammar constrained decoding failed, falling back to cursor "
               "implementation: "
            << e.what();
      }
    }

    if (handled_by_xgrammar) {
      continue;
    }

    if (i < root_cursors_.size() && !root_cursors_[i]) {
      root_cursors_[i] = build_root_cursor_for_sequence(i);
    }
    apply_cursor_constraint(i, generated_ids);
  }

  return any_constrained ? mask : torch::Tensor();
}

}  // namespace xllm
