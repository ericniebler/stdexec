/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <exec/sender_for.hpp>

#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>

namespace ex = STDEXEC;

namespace
{
  // Return a different sender when we invoke this custom defined let_value implementation
  struct let_value_test_domain
  {
    template <exec::sender_for<ex::let_value_t> Sender>
    static auto transform_sender(STDEXEC::set_value_t, Sender&&, auto&&...)
    {
      return ex::just(std::string{"hallo"});
    }
  };

  TEST_CASE("let_value can be customized", "[adaptors][let_value]")
  {
    basic_inline_scheduler<let_value_test_domain> sched;

    // The customization will return a different value
    auto snd = ex::just(std::string{"hello"}) | ex::continues_on(sched)
             | ex::let_value([](std::string& x) { return ex::just(x + ", world"); });
    using domain_t = ex::__completion_domain_of_t<ex::set_value_t, decltype(snd), ex::__sync_wait::__env>;
    static_assert(ex::__same_as<domain_t, let_value_test_domain>);
    //wait_for_value(std::move(snd), std::string{"hallo"});
  }
}  // namespace
