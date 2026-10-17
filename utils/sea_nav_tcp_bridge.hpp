#pragma once

#include "common_types.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <arpa/inet.h>
#include <cerrno>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using types::Vec3f;

// Bidirectional newline-delimited JSON link between the motion and perception
// hosts.  The bridge owns all socket operations in one thread; controller
// methods only exchange small snapshots protected by state_mutex_.
struct SeaNavCommandSnapshot {
    Vec3f velocity = Vec3f::Zero();
    bool received = false;
    bool fresh = false;
    bool enable = false;
    bool emergency_stop = false;
};

class SeaNavTcpBridge {
public:
    SeaNavTcpBridge(std::string bind_address = "192.168.1.102",
                    uint16_t port = 29001,
                    double command_timeout_s = 0.25,
                    double state_period_s = 0.02)
        : bind_address_(std::move(bind_address)),
          port_(port),
          command_timeout_s_(command_timeout_s),
          state_period_s_(state_period_s) {}

    ~SeaNavTcpBridge() { Stop(); }

    SeaNavTcpBridge(const SeaNavTcpBridge&) = delete;
    SeaNavTcpBridge& operator=(const SeaNavTcpBridge&) = delete;

    bool Start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return false;
        thread_ = std::thread(&SeaNavTcpBridge::ServerLoop, this);
        return true;
    }

    void Stop() {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }

    void SetMotionState(const Vec3f& projected_gravity,
                        const Vec3f& last_nav_command,
                        bool enabled, bool standing) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        projected_gravity_ = projected_gravity;
        last_nav_command_ = last_nav_command;
        enabled_ = enabled;
        standing_ = standing;
    }

    // A command received in manual mode must not start navigation later.
    void ClearCommand() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        nav_command_.setZero();
        command_received_ = false;
        command_enable_ = false;
        command_emergency_stop_ = false;
    }

    void SetAppliedNavCommand(const Vec3f& command) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        last_nav_command_ = command;
    }

    Vec3f GetAppliedNavCommand() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return last_nav_command_;
    }

    SeaNavCommandSnapshot GetCommand() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        SeaNavCommandSnapshot result;
        result.velocity = nav_command_;
        result.received = command_received_;
        result.enable = command_enable_;
        result.emergency_stop = command_emergency_stop_;
        result.fresh = command_received_ &&
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - command_time_).count() <=
                           command_timeout_s_;
        return result;
    }

    bool IsEmergencyStopActive() const {
        const auto command = GetCommand();
        return command.fresh && command.emergency_stop;
    }

    bool IsConnected() const {
        return connected_.load(std::memory_order_relaxed);
    }

private:
    static bool IsFinite(const nlohmann::json& value) {
        return value.is_number() && std::isfinite(value.get<double>());
    }

    static float Clamp(float value, float low, float high) {
        return std::max(low, std::min(value, high));
    }

    void CloseClient(int& client_fd) {
        if (client_fd >= 0) close(client_fd);
        client_fd = -1;
        connected_.store(false, std::memory_order_relaxed);
        receive_buffer_.clear();
    }

    void HandlePacket(const std::string& line) {
        try {
            const auto packet = nlohmann::json::parse(line);
            if (!packet.is_object() || packet.value("version", 0) != 1 ||
                packet.value("type", "") != "nav_command") {
                return;
            }
            if (!packet.contains("vx") || !packet.contains("vy") ||
                !packet.contains("vyaw") || !IsFinite(packet["vx"]) ||
                !IsFinite(packet["vy"]) || !IsFinite(packet["vyaw"])) {
                return;
            }
            const bool emergency_stop = packet.value("emergency_stop", false);
            const bool enable = packet.value("enable", true);
            const Vec3f command(
                Clamp(packet["vx"].get<float>(), -0.5f, 0.5f),
                Clamp(packet["vy"].get<float>(), -0.5f, 0.5f),
                Clamp(packet["vyaw"].get<float>(), -0.8f, 0.8f));
            std::lock_guard<std::mutex> lock(state_mutex_);
            nav_command_ = command;
            command_enable_ = enable;
            command_emergency_stop_ = emergency_stop;
            command_received_ = true;
            command_time_ = std::chrono::steady_clock::now();
        } catch (const std::exception&) {
            // Ignore malformed frames and keep the last valid command until
            // its timeout expires.
        }
    }

    std::string BuildMotionStatePacket() const {
        Vec3f gravity;
        Vec3f command;
        bool enabled;
        bool standing;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            gravity = projected_gravity_;
            command = last_nav_command_;
            enabled = enabled_;
            standing = standing_;
        }
        nlohmann::json packet = {
            {"version", 1},
            {"type", "motion_state"},
            {"stamp", std::chrono::duration<double>(
                           std::chrono::system_clock::now().time_since_epoch()).count()},
            {"projected_gravity", {gravity.x(), gravity.y(), gravity.z()}},
            {"last_nav_command", {command.x(), command.y(), command.z()}},
            {"enabled", enabled},
            {"standing", standing},
        };
        return packet.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n";
    }

    int CreateListenSocket() const {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port_);
        if (bind_address_ == "0.0.0.0" || bind_address_.empty()) {
            address.sin_addr.s_addr = htonl(INADDR_ANY);
        } else if (inet_pton(AF_INET, bind_address_.c_str(), &address.sin_addr) != 1) {
            close(fd);
            return -1;
        }
        if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
            listen(fd, 1) < 0) {
            close(fd);
            return -1;
        }
        return fd;
    }

    void ServerLoop() {
        int listen_fd = -1;
        int client_fd = -1;
        auto next_state = std::chrono::steady_clock::now();
        while (running_.load(std::memory_order_relaxed)) {
            if (listen_fd < 0) {
                listen_fd = CreateListenSocket();
                if (listen_fd < 0) {
                    std::cerr << "[SEA-Nav TCP] cannot listen on " << bind_address_
                              << ":" << port_ << ", retrying" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
                std::cout << "[SEA-Nav TCP] listening on " << bind_address_ << ":"
                          << port_ << std::endl;
            }

            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(listen_fd, &read_set);
            int max_fd = listen_fd;
            if (client_fd >= 0) {
                FD_SET(client_fd, &read_set);
                max_fd = std::max(max_fd, client_fd);
            }
            timeval timeout{0, 20000};
            const int ready = select(max_fd + 1, &read_set, nullptr, nullptr, &timeout);
            if (ready < 0) {
                if (errno != EINTR) {
                    CloseClient(client_fd);
                    close(listen_fd);
                    listen_fd = -1;
                }
                continue;
            }
            if (FD_ISSET(listen_fd, &read_set)) {
                int new_client = accept(listen_fd, nullptr, nullptr);
                if (new_client >= 0) {
                    CloseClient(client_fd);
                    client_fd = new_client;
                    connected_.store(true, std::memory_order_relaxed);
                    std::cout << "[SEA-Nav TCP] perception host connected" << std::endl;
                }
            }
            if (client_fd >= 0 && FD_ISSET(client_fd, &read_set)) {
                char buffer[4096];
                const ssize_t count = recv(client_fd, buffer, sizeof(buffer), 0);
                if (count <= 0) {
                    CloseClient(client_fd);
                } else {
                    receive_buffer_.append(buffer, static_cast<size_t>(count));
                    size_t newline = 0;
                    while ((newline = receive_buffer_.find('\n')) != std::string::npos) {
                        const std::string line = receive_buffer_.substr(0, newline);
                        receive_buffer_.erase(0, newline + 1);
                        if (!line.empty()) HandlePacket(line);
                    }
                    if (receive_buffer_.size() > 65536) receive_buffer_.clear();
                }
            }

            if (client_fd >= 0 && std::chrono::steady_clock::now() >= next_state) {
                const std::string packet = BuildMotionStatePacket();
                const ssize_t sent = send(client_fd, packet.data(), packet.size(), MSG_NOSIGNAL);
                if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                    CloseClient(client_fd);
                }
                next_state = std::chrono::steady_clock::now() +
                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(state_period_s_));
            }
        }
        CloseClient(client_fd);
        if (listen_fd >= 0) close(listen_fd);
        connected_.store(false, std::memory_order_relaxed);
    }

    std::string bind_address_;
    uint16_t port_;
    double command_timeout_s_;
    double state_period_s_;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::thread thread_;

    mutable std::mutex state_mutex_;
    Vec3f projected_gravity_ = Vec3f(0.f, 0.f, -1.f);
    Vec3f last_nav_command_ = Vec3f::Zero();
    Vec3f nav_command_ = Vec3f::Zero();
    bool enabled_ = false;
    bool standing_ = false;
    bool command_enable_ = false;
    bool command_emergency_stop_ = false;
    bool command_received_ = false;
    std::chrono::steady_clock::time_point command_time_ =
        std::chrono::steady_clock::time_point::min();
    std::string receive_buffer_;
};
