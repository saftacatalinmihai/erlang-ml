-module(neuron).

-export([hyp/2, perceive/2, rand_weights/1]).

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).

perceive(Inp, Weights) ->
    hyp([1|Inp], Weights).

%% Hard coded some vals, should be fine...
rand_weights(Number) -> rand_weights(Number, 0.12).
rand_weights(Number, Epsilon_init) ->
    [rand:uniform(100)/ 100 * 2 * Epsilon_init - Epsilon_init || _ <- lists:seq(1, Number)].

back_prop_delta(output, Actual, Predicted) ->
     Predicted - Actual;

back_prop_delta(hidden, Activation, WeightsOut, DeltaNextLayer) ->
    ml_math:hyp(WeightsOut, DeltaNextLayer).                      %% * ml_math:sigmoid_grad(Activation).


back_prop_deriv(Activation, DeltaNextLayer) = Activation * DeltaNextLayer