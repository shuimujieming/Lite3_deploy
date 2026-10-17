// Runs the real controller against a fake robot/policy and a loopback TCP link.
#include "rl_control_state_onnx.hpp"
#include <functional>
#include <stdexcept>

MotionStateFeedback StateBase::msfb_;

static void Check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void WaitFor(const std::function<bool()>& condition) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!condition()) {
        Check(std::chrono::steady_clock::now() < deadline, "TCP wait timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

class FakeRobot : public RobotInterface {
public:
    Vec3f rpy = Vec3f::Zero();
    FakeRobot() : RobotInterface("offline", 12) {}
    void Start() override {}
    void Stop() override {}
    double GetInterfaceTimeStamp() override { return 0; }
    VecXf GetJointPosition() override { return VecXf::Zero(12); }
    VecXf GetJointVelocity() override { return VecXf::Zero(12); }
    VecXf GetJointTorque() override { return VecXf::Zero(12); }
    Vec3f GetImuRpy() override { return rpy; }
    Vec3f GetImuAcc() override { return Vec3f::Zero(); }
    Vec3f GetImuOmega() override { return Vec3f::Zero(); }
    VecXf GetContactForce() override { return VecXf::Zero(4); }
    void SetJointCommand(Eigen::Matrix<float, Eigen::Dynamic, 5>) override {}
};

class FakeUser : public UserCommandInterface {
public:
    UserCommand command{};
    void Start() override {}
    void Stop() override {}
    UserCommand GetUserCommand() override { return command; }
    void SetMotionStateFeedback(const MotionStateFeedback& feedback) override {
        command.target_mode = feedback.current_state;
    }
};

class FakePolicy : public PolicyRunnerBase {
public:
    FakePolicy() : PolicyRunnerBase("offline") { decimation_ = 1; }
    void DisplayPolicyInfo() override {}
    void OnEnter() override {}
    RobotAction GetRobotAction(const RobotBasicState&) override {
        RobotAction action;
        action.goal_joint_pos = action.goal_joint_vel = action.kp = action.kd =
            action.tau_ff = VecXf::Zero(12);
        return action;
    }
};

// Reserve a free loopback port before creating the bridge (no robot connection).
static uint16_t FreePort() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    Check(fd >= 0, "cannot create loopback socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    Check(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "cannot reserve loopback port");
    socklen_t length = sizeof(address);
    Check(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) == 0,
          "cannot get loopback port");
    close(fd);
    return ntohs(address.sin_port);
}

int main() {
    auto data = std::make_shared<ControllerData>();
    auto robot = std::make_shared<FakeRobot>();
    auto user = std::make_shared<FakeUser>();
    const uint16_t port = FreePort();
    auto bridge = std::make_shared<SeaNavTcpBridge>("127.0.0.1", port, 0.05, 0.01);
    data->ri_ptr = robot;
    data->uc_ptr = user;
    data->ds_ptr = std::make_shared<DataStreaming>(false, false);
    data->sea_nav_ptr = bridge;
    RLControlStateONNX controller(Lite3, "rl_control", data, std::make_shared<FakePolicy>());
    bridge->Start();
    const int client = socket(AF_INET, SOCK_STREAM, 0);
    Check(client >= 0, "cannot create TCP client");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    WaitFor([&] { return connect(client, reinterpret_cast<sockaddr*>(&address),
                                sizeof(address)) == 0; });
    WaitFor([&] { return bridge->IsConnected(); });
    auto send_command = [&](float vx, bool enable, bool emergency,
                            float vy = 0.f, float vyaw = 0.f) {
        bridge->ClearCommand();
        const nlohmann::json packet = {{"version", 1}, {"type", "nav_command"},
            {"vx", vx}, {"vy", vy}, {"vyaw", vyaw}, {"enable", enable},
            {"emergency_stop", emergency}};
        const std::string wire = packet.dump() + "\n";
        Check(send(client, wire.data(), wire.size(), MSG_NOSIGNAL) ==
              static_cast<ssize_t>(wire.size()), "cannot send test command");
        WaitFor([&] { return bridge->GetCommand().received; });
    };
    auto near = [&](float value) {
        return std::abs(bridge->GetAppliedNavCommand().x() - value) < 1e-5f;
    };

    controller.SetNavigationMode(false);
    controller.OnEnter();
    Check(user->command.target_mode == RLControlMode, "manual feedback wrong");
    user->command.forward_vel_scale = 0.4f; // Manual target is 0.2 m/s.
    send_command(-0.5f, true, false);
    controller.Run();
    Check(near(0.1f), "TCP velocity took over manual RL");
    send_command(0.f, false, false);
    controller.Run();
    Check(near(0.15f), "disabled TCP stopped manual RL");
    send_command(0.f, false, true);
    Check(!controller.LoseControlJudge(), "remote emergency affected manual RL");
    user->command.target_mode = NavigationMode;
    Check(controller.GetNextStateName() == kNavigation, "cannot enter navigation");
    controller.OnExit();

    controller.SetNavigationMode(true);
    controller.OnEnter();
    Check(user->command.target_mode == NavigationMode, "navigation feedback wrong");
    Check(!bridge->GetCommand().received, "old TCP command survived mode entry");
    user->command.forward_vel_scale = 1.f;
    controller.Run();
    Check(near(0.f), "navigation used joystick without TCP");
    send_command(0.8f, true, false);
    controller.Run();
    Check(near(0.25f), "navigation did not use filtered TCP velocity");
    send_command(0.8f, false, false);
    controller.Run();
    Check(near(0.125f), "disabled navigation did not decelerate");
    send_command(0.8f, true, false);
    WaitFor([&] { return !bridge->GetCommand().fresh; });
    controller.Run();
    Check(near(0.0625f), "stale navigation did not decelerate");
    for (const float sign : {1.f, -1.f}) {
        send_command(3.f * sign, true, false, 3.f * sign, 3.f * sign);
        const Vec3f bounded(0.5f * sign, 0.5f * sign, 0.8f * sign);
        Check((bridge->GetCommand().velocity - bounded).norm() < 1e-5f,
              "TCP command exceeded velocity limits");
        const Vec3f previous = bridge->GetAppliedNavCommand();
        controller.Run();
        Check((bridge->GetAppliedNavCommand() - 0.5f * (bounded + previous)).norm() < 1e-5f,
              "controller velocity limit/filter mismatch");
    }
    send_command(0.f, false, true);
    Check(controller.LoseControlJudge(), "navigation emergency was ignored");
    bridge->ClearCommand();
    user->command.target_mode = JointDamping;
    Check(controller.LoseControlJudge(), "local navigation damping was ignored");
    user->command.target_mode = LieDown;
    Check(controller.GetNextStateName() == kLieDown, "navigation cannot lie down");
    user->command.target_mode = RLControlMode;
    Check(controller.GetNextStateName() == kRLControl, "navigation cannot return to RL");
    robot->rpy.x() = 0.6f;
    Check(controller.LoseControlJudge(), "navigation posture protection missing");
    robot->rpy.setZero();
    controller.OnExit();

    controller.SetNavigationMode(false);
    controller.OnEnter();
    controller.Run();
    Check(near(0.25f), "joystick did not recover after leaving navigation");
    user->command.side_vel_scale = 1.f;
    user->command.turnning_vel_scale = 1.f;
    controller.Run();
    const Vec3f manual_filtered(0.375f, 0.25f, 0.4f);
    Check((bridge->GetAppliedNavCommand() - manual_filtered).norm() < 1e-5f,
          "manual RL velocity limit/filter mismatch");
    user->command.target_mode = JointDamping;
    Check(controller.LoseControlJudge(), "local manual damping was ignored");
    user->command.target_mode = LieDown;
    Check(controller.GetNextStateName() == kLieDown, "manual cannot lie down");
    controller.OnExit();
    close(client);
    bridge->Stop();
    std::cout << "Navigation mode regression passed" << std::endl;
}
