%%%-------------------------------------------------------------------
%%% @author mihai
%%% @copyright (C) 2016, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 06. Sep 2016 2:51 PM
%%%-------------------------------------------------------------------
-module(neuron_funs).
-author("mihai").

%% API
-export([forward/5, backprop/4, test/0]).

%% runs feed forward on a single neuron.
%% returns the weighted sum of inputs and the activation function
%% as well as function refs to run the forwarding and get the gradient when backprop is received
forward(Inputs, Bias, Weights, Activation_fn, Deriv_fn) ->
  Z = ml_math:dot_p([Bias| Weights], [1| Inputs]),
  A = Activation_fn(Z),

  Gradient = fun (BackProp) ->
    BiasGrad = BackProp * Deriv_fn(Z),
    WeightsGrad =
      lists:map(
        fun(I) ->
          BackProp * Deriv_fn(Z) * I
        end,
        Inputs
      ),
    {BiasGrad, WeightsGrad}
  end,

  {Z, A, Gradient}.

backprop(Bias, Weights, Lambda, Gradients) ->
  {BiasGradSum, WeightsGradSum} =
    lists:foldl(
      fun({Bg, Wg}, {BgSum, Wgs}) ->
        {
          BgSum + Bg,
          lists:zipwith(fun(A, B) -> A + B end, Wg, Wgs)
        }
      end,
      {0, lists:map(fun(_) -> 0 end, lists:seq(1, length(Weights)))},
      Gradients
    ),

    NewBias = Bias + (Lambda / length(Gradients)) * BiasGradSum,
    NewWeights =
      lists:zipwith(
        fun(W, Wg) ->
          W + (Lambda / length(Gradients)) * Wg
        end,
        Weights, WeightsGradSum
      ),

  {NewBias, NewWeights}.

test() ->
  {_,A1,_,G1} = neuron_funs:forward(
    [1,1],
    0.12,
    [0.1, -0.04],
    fun ml_math:sigmoid/1,
    fun ml_math:sigmoid_deriv/1
%%    [o1,o2],
%%    fun (O,A) -> io:format("Output ~p received ~p~n", [O,A]) end
  ),

  {_,A2,_,G2} = neuron_funs:forward(
    [1,0],
    0.12,
    [0.1, -0.04],
    fun ml_math:sigmoid/1,
    fun ml_math:sigmoid_deriv/1
%%    [o1,o2],
%%    fun (O,A) -> io:format("Output ~p received ~p~n", [O,A]) end
  ),

  neuron_funs:backprop(
    0.12,
    [0.1, -0.04],
    1,
    [G1(1-A1),G2(0-A2)]
  ).
