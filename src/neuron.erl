%%%-------------------------------------------------------------------
%%% @author mihai
%%% @copyright (C) 2016, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 18. Sep 2016 19:11
%%%-------------------------------------------------------------------
-module(neuron).
-author("mihai").

-export([
  new/0,
  add_input/3,
  add_outputs/2,
  set_input_value/3,
  set_input_weight/3,
  get_input_weight/2,
  set_input_bias/2,
  activation/1,
  calc_activation/1,
  update_weights/2,
  input_Ids/1,
  output_Ids/1,
  all_inputs_set/1
]).

-export([test_new_neuron/0,test_3_neurons/0]).

-record(neuron, {
  inputs = [],
  outputs = [],
  bias = 0,
  activation = 0,
  activation_fn = fun ml_math:sigmoid/1,
  deriv_fn = fun ml_math:sigmoid_deriv/1,
  learning_rate = 1
}).

%% API
new() -> #neuron{}.

add_input(Neuron, Input_Id, Input_Value) ->
    Neuron#neuron{
      inputs = [{Input_Id, random_weight(), Input_Value, false} | Neuron#neuron.inputs]
    }.

add_outputs(Neuron, OutputIds) ->
  Neuron#neuron{
    outputs = OutputIds ++ Neuron#neuron.outputs
  }.

set_input_value(Neuron, Input_Id, Input_Value) ->
  {_,W,_,_} = lists:keyfind( Input_Id, 1, Neuron#neuron.inputs),

  Neuron#neuron{
    inputs = lists:keyreplace(
      Input_Id,
      1,
      Neuron#neuron.inputs,
      { Input_Id,
        W,
        Input_Value,
        true
      }
    )
  }.

set_input_bias(Neuron, Bias) -> Neuron#neuron{bias = Bias}.

set_input_weight(Neuron, Input_Id, Weight) ->
  {_,_,V,IS} = lists:keyfind( Input_Id, 1, Neuron#neuron.inputs),

  Neuron#neuron{
    inputs = lists:keyreplace(
      Input_Id,
      1,
      Neuron#neuron.inputs,
      { Input_Id,
        Weight,
        V,
        IS
      }
    )
  }.

get_input_weight(Neuron, Input_Id) ->
  {Input_Id,W,_V,_IS} = lists:keyfind( Input_Id, 1, Neuron#neuron.inputs),
  W.

calc_activation(Neuron) ->
  Z = ml_math:dot_p([Neuron#neuron.bias|input_weights(Neuron#neuron.inputs)], [1|input_values(Neuron#neuron.inputs)]),
  Neuron#neuron{
    activation = (Neuron#neuron.activation_fn)(Z),
    inputs = unset_all_inputs(Neuron#neuron.inputs)
  }.

activation(#neuron{activation = A}) -> A.

update_weights(Neuron, BackProp) ->
  Grads = neuron_funs:gradient(
    BackProp,
    Neuron#neuron.bias,
    input_weights(Neuron#neuron.inputs),
    input_values(Neuron#neuron.inputs),
    Neuron#neuron.deriv_fn
  ),
%%  io:format("!!!~p~n", [Grads]),
  {NewBias, NewWeights} = neuron_funs:backprop(
    Neuron#neuron.bias,
    input_weights(Neuron#neuron.inputs),
    Neuron#neuron.learning_rate,
    [Grads]
    ),
  Neuron#neuron{
    bias = NewBias,
    inputs =  lists:map(
      fun ({{Input_Id, _, Input_Value, Input_Set}, NW}) ->
        {Input_Id, NW, Input_Value, Input_Set}
      end,
      lists:zip(Neuron#neuron.inputs, NewWeights)
    )
  }.

all_inputs_set(Neuron) ->
  lists:all(fun({_Id, _W, _V, IS}) -> IS =:= true end, Neuron#neuron.inputs).

unset_all_inputs(Inputs) ->
  lists:map(fun({Id, W, V, _IS}) -> {Id, W, V, false} end, Inputs).

%% Internal functions
random_weight() -> rand:uniform() * 0.12 * 2 - 0.12.
input_values(Inputs) ->
  lists:map(fun({_Id, _W, V, _IS}) -> V end, Inputs).

input_weights(Inputs) ->
  lists:map(fun({_Id, W, _V, _IS}) -> W end, Inputs).

input_Ids (Neuron) ->
  lists:map(fun({Id, _W, _V, _IS}) -> Id end,  Neuron#neuron.inputs).


output_Ids (#neuron{outputs = O}) -> O.


% Test
test_new_neuron() ->
  N = new(),
  N1 = add_input(N, 1, 1),
  N2 = add_input(N1, 2, 1),
  io:format("~p~n", [N2]),

  N3 = calc_activation(N2),
  N4 = update_weights(N3, 0 - activation(N3)),
  io:format("~p~n", [N4]).

test_3_neurons() ->
  N1 = new(),
  N11 = add_input(N1, 1, 1),
  N12 = add_input(N11, 2, 1),
  N13 = calc_activation(N12),

  N2 = new(),
  N21 = add_input(N2, 1, 1),
  N22 = add_input(N21, 2, 1),
  N23 = calc_activation(N22),

  N3 = new(),
  N31 = add_input(N3, 1, 0),
  N32 = add_input(N31, 2, 0),
  N33 = set_input_value(N32, 1, activation(N13)),
  N34 = set_input_value(N33, 2, activation(N23)),
  N35 = calc_activation(N34),

  io:format("~p~n", [N35]).