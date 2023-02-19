#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>
#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"

namespace ex = stdexec;
using exec::async_scope_context;
using stdexec::sync_wait;

TEST_CASE("empty will complete immediately on an empty async_scope_context", "[async_scope_context][empty]") {
  async_scope_context context;
  bool is_empty{false};

  ex::sender auto snd = exec::async_resource.close(context) | ex::then([&] { is_empty = true; });
  sync_wait(std::move(snd));
  REQUIRE(is_empty);
}

TEST_CASE("empty sender can properly connect a void receiver", "[async_scope_context][empty]") {
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);
  bool is_empty{false};

  exec::async_scope.spawn(scope, ex::just());

  ex::sender auto snd = exec::async_resource.close(context) | ex::then([&] { is_empty = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  REQUIRE(is_empty);
}

TEST_CASE("empty will complete after the work is done", "[async_scope_context][empty]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // Add some work
  exec::async_scope.spawn(scope, ex::on(sch, ex::just()));

  // The on_empty() sender cannot notify now
  bool is_empty{false};
  ex::sender auto snd = ex::on(sch, exec::async_resource.close(context)) | ex::then([&] { is_empty = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  REQUIRE_FALSE(is_empty);

  // TODO: refactor this test
  sch.start_next();
  sch.start_next();
  // We should be notified now
  REQUIRE(is_empty);
}

TEST_CASE("TODO: empty can be used multiple times", "[async_scope_context][empty]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // Add some work
  exec::async_scope.spawn(scope, ex::on(sch, ex::just()));

  // The on_empty() sender cannot notify now
  bool is_empty{false};
  ex::sender auto snd = ex::on(sch, exec::async_resource.close(context)) | ex::then([&] { is_empty = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  REQUIRE_FALSE(is_empty);

  // TODO: refactor this test
  sch.start_next();
  sch.start_next();
  // We should be notified now
  REQUIRE(is_empty);

  // Add some work
  exec::async_scope.spawn(scope, ex::on(sch, ex::just()));

  // The on_empty() sender cannot notify now
  bool is_empty2{false};
  ex::sender auto snd2 = ex::on(sch, exec::async_resource.close(context)) | ex::then([&] { is_empty2 = true; });
  auto op2 = ex::connect(std::move(snd2), expect_void_receiver{});
  ex::start(op2);
  REQUIRE_FALSE(is_empty2);

  // TODO: refactor this test
  sch.start_next();
  sch.start_next();
  // We should be notified now
  REQUIRE(is_empty2);
}

// TODO: GCC-11 generates warnings (treated as errors) for the following test
#if defined(__clang__) || !defined(__GNUC__)
TEST_CASE("waiting on work that spawns more work", "[async_scope_context][empty]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  bool work1_done{false};
  auto work1 = [&] {
    work1_done = true;
  };
  bool work2_done{false};
  auto work2 = [&] {
    // Spawn work
    exec::async_scope.spawn(scope, ex::on(sch, ex::just() | ex::then(work1)));
    // We are done
    work2_done = true;
  };

  // Spawn work 2
  // No work is executed until the impulse scheduler dictates
  exec::async_scope.spawn(scope, ex::on(sch, ex::just() | ex::then(work2)));

  // start an on_empty() sender
  bool is_empty{false};
  ex::sender auto snd = ex::on(inline_scheduler{}, exec::async_resource.close(context)) //
    | ex::then([&] { is_empty = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  REQUIRE_FALSE(work1_done);
  REQUIRE_FALSE(work2_done);
  REQUIRE_FALSE(is_empty);

  // Trigger the execution of work2
  // When work2 is done, work1 is not yet started
  sch.start_next();
  REQUIRE_FALSE(work1_done);
  REQUIRE(work2_done);
  REQUIRE_FALSE(is_empty);

  // Trigger the execution of work1
  // This will complete the on_empty() sender
  sch.start_next();
  REQUIRE(work1_done);
  REQUIRE(work2_done);
  REQUIRE(is_empty);
}
#endif

// TODO: GCC-11 generates warnings (treated as errors) for the following test
#if defined(__clang__) || !defined(__GNUC__)
TEST_CASE(
  "async_scope_context is empty after adding work when in cancelled state",
  "[async_scope_context][empty]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  bool is_empty1{false};
  ex::sender auto snd = ex::on(inline_scheduler{}, exec::async_resource.close(context)) //
    | ex::then([&] { is_empty1 = true; });
  auto op = ex::connect(std::move(snd), expect_void_receiver{});
  ex::start(op);
  REQUIRE(is_empty1);

  // cancel & add work
  context.request_stop();
  bool work_executed{false};
  exec::async_scope.spawn(scope, ex::on(sch, ex::just())
    | ex::upon_stopped([&] { work_executed = true; printf(".\n");}));
  // note that we don't tell impulse sender to start the work

  bool is_empty2{false};
  ex::sender auto snd2 = ex::on(inline_scheduler{}, exec::async_resource.close(context)) //
    | ex::then([&] { is_empty2 = true; });
  auto op2 = ex::connect(std::move(snd2), expect_void_receiver{});
  ex::start(op2);
  REQUIRE_FALSE(is_empty2);

  REQUIRE_FALSE(work_executed);
  sch.start_next();
  REQUIRE(work_executed);
  REQUIRE(is_empty2);
}
#endif
