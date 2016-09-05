-module(ml_math_tests).

-include_lib("eunit/include/eunit.hrl").

sigmoid_test_() ->
    [
        fun () -> ?assert(ml_math:sigmoid(0) =:= 0.5   ) end,
        fun () -> ?assert(ml_math:sigmoid(-10) < 0.001 ) end,
        fun () -> ?assert(ml_math:sigmoid(10)  > 0.001 ) end
    ].

hyp_test_() ->
    fun() -> ?assert( ml_math:hyp([1,2],[2,3], fun ml_math:sigmoid/1) =:= 0.9996646498695336 ) end.

gradient_descent_lin_reg_test_()->
    W = ml_math:gradient_descent(
        [0,0],
        [[2], [3], [4]],
        [4,6,8],
        0.1,
        fun ml_math:lin_reg_cost/3,
        fun ml_math:lin_reg_deriv/4,
        1500
    ),
    [
        fun () ->
            H = ml_math:dot_p(W, [1, 10]),
            ?assert(H > 19.999),
            ?assert(H < 20.001)
         end
    ].

sigmoid_grad_test_() ->
    [fun () -> ?assert( ml_math:sigmoid_deriv(2) =:= 0.10499358540350662) end ].

