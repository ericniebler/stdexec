#include <catch2/catch.hpp>
#include <exec/async_scope.hpp>
#include "test_common/schedulers.hpp"
#include "test_common/receivers.hpp"
#include "test_common/type_helpers.hpp"

namespace ex = stdexec;
using exec::async_scope_context;
using stdexec::sync_wait;

//! Sender that throws exception when connected
struct throwing_sender {
  using is_sender = void;
  using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

  template <class Receiver>
  struct operation {
    Receiver rcvr_;

    friend void tag_invoke(ex::start_t, operation& self) noexcept {
      ex::set_value(std::move(self.rcvr_));
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, throwing_sender&& self, Receiver&& rcvr)
    -> operation<std::decay_t<Receiver>> {
    throw std::logic_error("cannot connect");
    return {std::forward<Receiver>(rcvr)};
  }

  friend empty_env tag_invoke(stdexec::get_env_t, const throwing_sender&) noexcept {
    return {};
  }
};

TEST_CASE("spawn will execute its work", "[async_scope_context][spawn]") {
  impulse_scheduler sch;
  bool executed{false};
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // Non-blocking call
  exec::async_scope.spawn(scope, ex::on(sch, ex::just() | ex::then([&] { executed = true; })));
  REQUIRE_FALSE(executed);
  // Run the operation on the scheduler
  sch.start_next();
  // Now the spawn work should be completed
  REQUIRE(executed);
}

TEST_CASE("spawn will start sender before returning", "[async_scope_context][spawn]") {
  bool executed{false};
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // This will be a blocking call
  exec::async_scope.spawn(scope, ex::just() | ex::then([&] { executed = true; }));
  REQUIRE(executed);
}

#if !NO_TESTS_WITH_EXCEPTIONS
TEST_CASE(
  "spawn will propagate exceptions encountered during op creation", 
  "[async_scope_context][spawn]") {
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);
  try {
    exec::async_scope.spawn(scope, throwing_sender{} | ex::then([&] { FAIL("work should not be executed"); }));
    FAIL("Exceptions should have been thrown");
  } catch (const std::logic_error& e) {
    SUCCEED("correct exception caught");
  } catch (...) {
    FAIL("invalid exception caught");
  }
}
#endif

TEST_CASE(
  "TODO: spawn will keep the scope non-empty until the work is executed",
  "[async_scope_context][spawn]") {
  impulse_scheduler sch;
  bool executed{false};
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // Before adding any operations, the scope is empty
  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));

  // Non-blocking call
  exec::async_scope.spawn(scope, ex::on(sch, ex::just() | ex::then([&] { executed = true; })));
  REQUIRE_FALSE(executed);

  // The scope is now non-empty
  // TODO: reenable this
  // REQUIRE_FALSE(P2519::__scope::empty(scope));
  // REQUIRE(P2519::__scope::op_count(scope) == 1);

  // Run the operation on the scheduler; blocking call
  sch.start_next();

  // Now the scope should again be empty
  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
  REQUIRE(executed);
}

TEST_CASE(
  "TODO: spawn will keep track on how many operations are in flight", 
  "[async_scope_context][spawn]") {
  impulse_scheduler sch;
  std::size_t num_executed{0};
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // Before adding any operations, the scope is empty
  // TODO: reenable this
  // REQUIRE(P2519::__scope::op_count(scope) == 0);
  // REQUIRE(P2519::__scope::empty(scope));

  constexpr std::size_t num_oper = 10;
  for (std::size_t i = 0; i < num_oper; i++) {
    exec::async_scope.spawn(scope, ex::on(sch, ex::just() | ex::then([&] { num_executed++; })));
    size_t num_expected_ops = i + 1;
    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
    (void) num_expected_ops;
  }

  // Now execute the operations
  for (std::size_t i = 0; i < num_oper; i++) {
    sch.start_next();
    size_t num_expected_ops = num_oper - i - 1;
    // TODO: reenable this
    // REQUIRE(P2519::__scope::op_count(scope) == num_expected_ops);
    (void) num_expected_ops;
  }

  // The scope is empty after all the operations are executed
  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
  REQUIRE(num_executed == num_oper);
}

TEST_CASE("TODO: spawn work can be cancelled by cancelling the scope", "[async_scope_context][spawn]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  bool cancelled1{false};
  bool cancelled2{false};

  exec::async_scope.spawn(
    scope, 
    ex::on(
      sch, 
      ex::just() 
        | ex::let_stopped([&] {
          cancelled1 = true;
          return ex::just();
        })));
  exec::async_scope.spawn(
    scope, 
    ex::on(
      sch, 
      ex::just() 
        | ex::let_stopped([&] {
          cancelled2 = true;
          return ex::just();
        })));

  // TODO: reenable this
  // REQUIRE(P2519::__scope::op_count(scope) == 2);

  // Execute the first operation, before cancelling
  sch.start_next();
  REQUIRE_FALSE(cancelled1);
  REQUIRE_FALSE(cancelled2);

  // Cancel the async_scope_context object
  context.request_stop();
  // TODO: reenable this
  // REQUIRE(P2519::__scope::op_count(scope) == 1);

  // Execute the first operation, after cancelling
  sch.start_next();
  REQUIRE_FALSE(cancelled1);
  // TODO: second operation should be cancelled
  // REQUIRE(cancelled2);
  REQUIRE_FALSE(cancelled2);

  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
}

template <typename S>
concept is_spawn_worthy = requires(S&& snd, exec::__scope::__async_scope& scope) { 
  exec::async_scope.spawn(scope, std::move(snd), ex::empty_env{}); 
};

TEST_CASE("spawn accepts void senders", "[async_scope_context][spawn]") {
  static_assert(is_spawn_worthy<decltype(ex::just())>);
}
TEST_CASE(
  "spawn doesn't accept non-void senders", 
  "[async_scope_context][spawn]") {
  static_assert(!is_spawn_worthy<decltype(ex::just(13))>);
  static_assert(!is_spawn_worthy<decltype(ex::just(3.14))>);
  static_assert(!is_spawn_worthy<decltype(ex::just("hello"))>);
}
TEST_CASE(
  "TODO: spawn doesn't accept senders of errors", 
  "[async_scope_context][spawn]") {
  // TODO: check if just_error(exception_ptr) should be allowed
  static_assert(is_spawn_worthy<decltype(ex::just_error(std::exception_ptr{}))>);
  static_assert(!is_spawn_worthy<decltype(ex::just_error(std::error_code{}))>);
  static_assert(!is_spawn_worthy<decltype(ex::just_error(-1))>);
}
TEST_CASE(
  "spawn should accept senders that send stopped signal", 
  "[async_scope_context][spawn]") {
  static_assert(is_spawn_worthy<decltype(ex::just_stopped())>);
}

TEST_CASE(
  "TODO: spawn works with senders that complete with stopped signal", 
  "[async_scope_context][spawn]") {
  impulse_scheduler sch;
  async_scope_context context;
  exec::satisfies<exec::async_scope> auto scope = exec::async_resource.get_resource_token(context);

  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));

  exec::async_scope.spawn(scope, ex::on(sch, ex::just_stopped()));

  // The scope is now non-empty
  // TODO: reenable this
  // REQUIRE_FALSE(P2519::__scope::empty(scope));
  // REQUIRE(P2519::__scope::op_count(scope) == 1);

  // Run the operation on the scheduler; blocking call
  sch.start_next();

  // Now the scope should again be empty
  // TODO: reenable this
  // REQUIRE(P2519::__scope::empty(scope));
}
