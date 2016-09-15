-module(ml_math).

-import(lists, [nth/2, seq/2]).

-compile(export_all).
%-export([sigmoid/1, hyp/3, hyp/2, gradient_descent/7, lin_reg_deriv/4, lin_reg_cost/3]).

sigmoid(Z) when Z > 10 ->  1;
sigmoid(Z) when Z < -10 -> 0;
sigmoid(Z) -> 1 / (1 + math:exp(-Z)).

sigmoid_deriv(Z) ->
    Gz = sigmoid(Z),
    Gz * (1 - Gz).

dot_p(W, A) ->
  lists:sum(
    lists:zipwith(
      fun (X,Y) -> X * Y end,
      W,A
    )
  ).

hyp(W, A, F) -> F(dot_p(W, A)).

gradient_descent(Weights, _, _, _, _, _, 0) -> Weights;
gradient_descent(Weights, X, Y, LearningRate, _CostF, DerivF, Iterations) ->
%    io:format("Cost: ~p~n", [lin_reg_cost(Weights,X,Y)]),
    gradient_descent(
        lists:map(
            fun (J) ->
                nth(J,Weights) - LearningRate * DerivF(Weights, X, Y, J)
            end,
            seq(1, length(Weights))
        ), X, Y, LearningRate, _CostF, DerivF, Iterations - 1
    ).

lin_reg_deriv(W, X, Y, J) ->
    lists:foldl(
        fun (I, Sum) ->
            Sum + (dot_p(W, [1 | nth(I, X)]) - nth(I, Y) ) * nth(J, [1 | nth(I, X)])
        end,
        0,
        seq(1,length(X))
    ) / length(X).

lin_reg_cost(W, X, Y) ->
    lists:foldl(
        fun(I, Sum) ->
            Diff = (dot_p(W, [1 | nth(I, X)]) - nth(I, Y) ),
            Sum + Diff * Diff
        end,
        0,
        seq(1, length(X))
    ) / 2 * length(X).