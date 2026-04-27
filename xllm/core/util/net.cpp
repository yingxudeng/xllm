/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "net.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace xllm {
namespace net {

static std::mutex g_port_mutex;
static std::unordered_set<int> g_allocated_port_map;

namespace {

constexpr char kIpv4Any[] = "0.0.0.0";
constexpr char kIpv6Any[] = "::";
constexpr char kIpv4Loopback[] = "127.0.0.1";
constexpr char kIpv6Loopback[] = "::1";
constexpr char kLocalhost[] = "localhost";

std::string extract_ip_for_local_check(const std::string& input) {
  if (input.empty()) {
    return "";
  }

  if (input.front() == '[') {
    size_t bracket_pos = input.find(']');
    if (bracket_pos != std::string::npos) {
      return input.substr(1, bracket_pos - 1);
    }
  }

  if (input.find(':') != input.rfind(':')) {
    return input;
  }

  return extract_ip(input);
}

bool is_wildcard_ip(const std::string& ip) {
  return ip == kIpv4Any || ip == kIpv6Any;
}

bool is_loopback_ip(const std::string& ip) {
  return ip == kIpv4Loopback || ip == kIpv6Loopback || ip == kLocalhost;
}

bool sockaddr_matches_ip(const sockaddr* addr, const std::string& ip) {
  if (addr == nullptr) {
    return false;
  }

  char host[INET6_ADDRSTRLEN]{'\0'};
  if (addr->sa_family == AF_INET) {
    const sockaddr_in* addr_in = reinterpret_cast<const sockaddr_in*>(addr);
    const char* result =
        inet_ntop(AF_INET, &addr_in->sin_addr, host, sizeof(host));
    return result != nullptr && ip == host;
  }

  if (addr->sa_family == AF_INET6) {
    const sockaddr_in6* addr_in6 = reinterpret_cast<const sockaddr_in6*>(addr);
    const char* result =
        inet_ntop(AF_INET6, &addr_in6->sin6_addr, host, sizeof(host));
    return result != nullptr && ip == host;
  }

  return false;
}

}  // namespace

// TODO: return private ip
std::string get_local_ip_addr() {
  char ip[INET_ADDRSTRLEN]{'\0'};
  char hostname[256];
  int ret = gethostname(hostname, sizeof(hostname));
  if (ret != 0) {
    LOG(ERROR) << "gethostname failed";
    return "";
  }
  struct addrinfo* info = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  ret = getaddrinfo(hostname, nullptr, &hints, &info);
  if (ret != 0) {
    LOG(ERROR) << "getaddrinfo failed";
    return "";
  }
  auto guard = std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)>(
      info, freeaddrinfo);
  auto* addr = (struct sockaddr_in*)info->ai_addr;
  auto* result = inet_ntop(addr->sin_family, &addr->sin_addr, ip, sizeof(ip));

  return std::string(ip);
}

int get_local_free_port() {
  std::lock_guard<std::mutex> lock(g_port_mutex);
  int port;
  do {
    port = 0;
    struct sockaddr_in addr;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr*)&addr, &len) == -1) {
      return -1;
    }
    port = ntohs(addr.sin_port);

    ::close(fd);
  } while (g_allocated_port_map.find(port) != g_allocated_port_map.end());

  g_allocated_port_map.insert(port);

  return port;
}

uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port) {
  in_addr ip_addr;
  CHECK(inet_pton(AF_INET, ip.c_str(), &ip_addr) == 1)
      << "Invalid IPv4 address format : " << ip;

  uint32_t ip_network = ip_addr.s_addr;
  return (static_cast<uint64_t>(ip_network) << 32) | port;
}

std::pair<std::string, uint16_t> convert_uint64_to_ip_port(uint64_t input) {
  uint16_t port = static_cast<uint16_t>(input & 0xFFFF);
  uint32_t ip_network = static_cast<uint32_t>(input >> 32);

  in_addr ip_addr;
  ip_addr.s_addr = ip_network;

  char ip_str[INET_ADDRSTRLEN];
  const char* result = inet_ntop(AF_INET, &ip_addr, ip_str, INET_ADDRSTRLEN);
  CHECK(result != nullptr) << "Failed to convert IP address from uint64: "
                           << input;

  return {std::string(ip_str), port};
}

// input example: 127.0.0.1:18889
std::string extract_ip(const std::string& input) {
  std::istringstream stream(input);
  std::string ip;

  std::getline(stream, ip, ':');
  if (ip == "127.0.0.1") {
    ip = get_local_ip_addr();
  }
  return ip;
}

std::string extract_port(const std::string& input) {
  std::istringstream stream(input);
  std::string ip;
  std::string port;

  std::getline(stream, ip, ':');
  std::getline(stream, port, ':');

  return port;
}

bool is_local_ip(const std::string& ip) {
  if (ip.empty() || is_wildcard_ip(ip)) {
    return false;
  }

  if (is_loopback_ip(ip)) {
    return true;
  }

  ifaddrs* ifaddr = nullptr;
  int ret = getifaddrs(&ifaddr);
  if (ret != 0) {
    LOG(WARNING) << "getifaddrs failed, fallback to hostname ip check";
    return ip == get_local_ip_addr();
  }

  auto guard =
      std::unique_ptr<ifaddrs, decltype(&freeifaddrs)>(ifaddr, freeifaddrs);
  for (ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (sockaddr_matches_ip(ifa->ifa_addr, ip)) {
      return true;
    }
  }

  return false;
}

bool is_local_peer_addr(const std::string& bind_addr,
                        const std::string& peer_addr) {
  const std::string bind_ip = extract_ip_for_local_check(bind_addr);
  const std::string peer_ip = extract_ip_for_local_check(peer_addr);

  if (bind_ip.empty() || peer_ip.empty()) {
    return false;
  }

  if (bind_ip == peer_ip) {
    return true;
  }

  if (is_wildcard_ip(bind_ip)) {
    return is_local_ip(peer_ip);
  }

  return false;
}

void parse_host_port_from_addr(const std::string& addr,
                               std::string& host,
                               int& port) {
  CHECK(!addr.empty()) << "Address is empty";

  auto colon_pos = addr.find(':');
  CHECK_NE(colon_pos, std::string::npos) << "Invalid address format: " << addr;

  host = addr.substr(0, colon_pos);
  port = std::stoi(addr.substr(colon_pos + 1));
}

}  // namespace net
}  // namespace xllm
