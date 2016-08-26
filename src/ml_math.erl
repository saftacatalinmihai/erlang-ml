-module(ml_math).

-import(lists, [nth/2, seq/2]).

-export([sigmoid/1, hyp/3, hyp/2, gradient_descent/7, lin_reg_deriv/4, lin_reg_cost/3]).

sigmoid(Z) -> 1 / (1 + math:exp(-Z)).

hyp(W, A) -> hyp_acc(W, A, 0).
hyp(W, A, F) -> F(hyp_acc(W, A, 0)).

hyp_acc([],[], Acc) -> Acc;
hyp_acc([W|WT], [A|AT], Acc) ->
    hyp_acc(WT, AT, Acc + W*A).

gradient_descent(Weights, _, _, _, _, _, 0) -> Weights;
gradient_descent(Weights, X, Y, Alpha, _CostF, DerivF, Iterations) ->
%    io:format("Cost: ~p~n", [lin_reg_cost(Weights,X,Y)]),
    gradient_descent(
        lists:map(
            fun (J) ->
                nth(J,Weights) - Alpha * DerivF(Weights, X, Y, J)
            end,
            seq(1, length(Weights))
        ), X, Y, Alpha, _CostF, DerivF, Iterations - 1
    ).

lin_reg_deriv(W, X, Y, J) ->
    lists:foldl(
        fun (I, Sum) ->
            Sum + (hyp(W, [1 | nth(I, X)]) - nth(I, Y) ) * nth(J, [1 | nth(I, X)])
        end,
        0,
        seq(1,length(X))
    ) / length(X).

lin_reg_cost(W, X, Y) ->
    lists:foldl(
        fun(I, Sum) ->
            Diff = (hyp(W, [1 | nth(I, X)]) - nth(I, Y) ),
            Sum + Diff * Diff
        end,
        0,
        seq(1, length(X))
    ) / 2 * length(X).