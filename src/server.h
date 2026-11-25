// OpenAI-style server over an Engine: POST /v1/completions and /v1/chat/completions (messages through the chat
// template; the reasoning before "</think>" comes as reasoning_content), streamed (SSE) or not, with stop strings;
// GET /v1/models. One engine thread owns the Engine (architecture.md rule 3); httplib's handler threads only queue
// requests and read their text. A client that disconnects cancels its request.
//
//   mllm::Server server(engine, "mlx-community/Qwen3.5-0.8B-4bit");
//   server.listen("127.0.0.1", 8080);  // blocks until server.stop() from another thread
#pragma once
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "engine.h"

namespace httplib {
class Server;
struct Request;
struct Response;
}

namespace mllm {

class Server {
public:
    Server(Engine& engine, std::string model_name);
    ~Server();
    bool listen(const std::string& host, int port);  // blocks until stop(); false if the port can't be bound
    void wait_until_ready() const;
    // Stops accepting and cuts the open streams (their requests are cancelled); listen() returns once the engine
    // thread is done.
    void stop();
    size_t live_requests();  // the engine's live requests, as of its last loop

    struct Stream;  // one request: its prompt in, its events out

private:
    void run_engine();  // the engine thread
    void handle_request(const httplib::Request& request, httplib::Response& response, bool chat);
    void queue_cancel(const std::shared_ptr<Stream>& stream);  // the engine thread cancels it on its next loop
    // The engine thread's steps: tokenize and submit new streams (or reject them), cancel the cancelled ones, hand
    // one engine event to its stream
    void start_streams(const std::deque<std::shared_ptr<Stream>>& streams);
    void cancel_streams(const std::deque<std::shared_ptr<Stream>>& streams);
    void deliver(const Event& event);

    Engine& engine;
    std::map<int, std::shared_ptr<Stream>> live_streams;  // engine thread only: engine request id -> its stream
    std::string model_name;
    std::unique_ptr<httplib::Server> http;
    std::mutex mutex;  // guards new_streams, cancelled_streams, stopping, live_count
    std::condition_variable wake_engine;
    std::deque<std::shared_ptr<Stream>> new_streams, cancelled_streams;
    bool stopping = false;
    size_t live_count = 0;
};

}  // namespace mllm
