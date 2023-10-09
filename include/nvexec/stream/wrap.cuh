/*
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
#pragma once

#include "../../stdexec/execution.hpp"
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace _wrap {
    template <class SenderId>
    struct sender : stream_sender_base {
      using is_sender = void;
      using Sender = stdexec::__t<SenderId>;
      using __t = sender;
      using __id = sender;

      sender(Sender sndr, context_state_t context_state)
        : sndr_(std::move(sndr))
        , env_{context_state} {
      }

      struct environment {
        context_state_t context_state_;

        template <same_as<environment> Self>
        friend auto tag_invoke(get_completion_scheduler_t<set_value_t>, const Self& env) noexcept {
          return env.context_state_.make_stream_scheduler();
        }
      };

      // BUGBUG this doesn't handle the case where the sender has a nested
      // type alias named completion_signatures.
      template <class Self, class Env>
      using completions_t =
        tag_invoke_result_t<get_completion_signatures_t, __copy_cvref_t<Self, Sender>, Env>;

      // test for tag_invocable instead of sender_to because the connect customization
      // point would convert the stdexec::just sender back into this nvexec::just sender,
      // causing recursion.
      template <__decays_to<sender> Self, receiver Receiver>
        requires receiver_of<Receiver, completions_t<Self, env_of_t<Receiver>>> &&
          tag_invocable<connect_t, __copy_cvref_t<Self, Sender>, Receiver>
      friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr) //
        noexcept(nothrow_tag_invocable<connect_t, __copy_cvref_t<Self, Sender>, Receiver>)
          -> tag_invoke_result_t<connect_t, __copy_cvref_t<Self, Sender>, Receiver> {
        return tag_invoke(connect, ((Self&&) self).sndr_, (Receiver&&) rcvr);
      }

      template <__decays_to<sender> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&& self, Env&& env) noexcept
        -> completions_t<Self, Env> {
        return {};
      }

      template <same_as<sender> Self>
      friend const environment& tag_invoke(get_env_t, const Self& self) noexcept {
        return self.env_;
      }

      Sender sndr_;
      environment env_;
    };
  } // namespace _wrap

  template <class Env, class Sender>
  auto as_stream_sender(Sender sndr, const context_state_t&) -> Sender {
    return sndr;
  }

  template <class Env, class Sender>
    requires _non_stream_sender<Sender, Env>
  auto as_stream_sender(Sender sndr, const context_state_t& context_state) //
    -> _wrap::sender<__id<Sender>> {
    return {std::move(sndr), context_state};
  }
}

namespace stdexec::__detail {
  template <class SenderId>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::_wrap::sender<__name_of<__t<SenderId>>>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::_wrap::sender<SenderId>>{};
}
