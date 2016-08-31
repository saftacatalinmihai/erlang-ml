-module(neuron).

%%-export([hyp/2, perceive/2, rand_weights/1]).
-compile(export_all).

-behavior(gen_server).

-record( input, {
  weight      = rand:uniform(), %% the weight associated to the input stimulus
  activation  = 0               %% the last activation of input node
}).

-record( output, {
  backprop = 0                  %% backprop = weight * delta
}).

-record( state, {
  inputs     = #{1 => #input{activation = 1}},  %% map between input Node and record. Starts with the bias term id = 1
  outputs    = #{},                             %% map between output Node and record
  activation = 0                                %% last activation of this node
}).

%% API -----------------------------------------------------------------------------------------------------------------
start_link() -> gen_server:start_link(?MODULE, [], []).

stimulate(Node, FromNode, Activation) -> gen_server:cast(Node, {stimulate, FromNode, Activation}).

%%learn(BackNode, Node, Delta) -> gen_server:cast(BackNode, {learn, {Node, Delta}}).

connect(Input_node, Output_Node) ->
  gen_server:cast(Input_node, {connect_to_output, Output_Node}),
  gen_server:cast(Output_Node, {connect_to_input, Input_node}).

pass(Pid, Input) -> gen_server:cast(Pid, {pass, Input}).

%% Debugging api
set_weights(Pid, W) -> gen_server:call(Pid, {set_weights, W }).

get_output(Pid) -> gen_server:call(Pid, get_output).

%%----------------------------------------------------------------------------------------------------------------------

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).

perceive(Inp, Weights) ->
    hyp(Inp, Weights).

init([]) ->
  {ok, #state{}}.

handle_cast({stimulate, FromNode, Activation}, State) ->
  #{FromNode := InputNode} = State#state.inputs,

  New_Inputs = maps:update(
    FromNode,
    InputNode#input{activation = Activation},
    State#state.inputs
  ),

  InputList = maps:values(New_Inputs),
  Neuron_Activation = perceive(
    lists:map(fun(I) -> I#input.activation end, InputList),
    lists:map(fun(I) -> I#input.weight     end, InputList)
  ),

  case maps:size(State#state.outputs) of
    0 ->
      io:format("~n~w Activation: ~w", [self(), Neuron_Activation]);
%%      neuron:learn(self(), self(), 1)
    _ ->
      lists:foreach(
        fun(Output_PID) ->
          neuron:stimulate(Output_PID, self(), Neuron_Activation)
        end,
        maps:keys(State#state.outputs)
      )
  end,
  {noreply, State#state{inputs = New_Inputs, activation = Neuron_Activation}};
%%
%%handle_cast({learn, Backprop}, #state{weights=Weights, input_nodes = Inputs, output_nodes = Outputs}) ->
%%
%%  Learning_rate = 0.5,
%%
%%  % Calculate the correct sensitivities
%%  io:format("Outputs: ~p~n", [Outputs]),
%%  io:format("Backprop: ~p~n", [Backprop]),
%%  New_outputs = add_delta(Outputs, Backprop),
%%  Output_value = perceive(convert_to_list(Inputs), Weights),
%%  Derv_value = ml_math:hyp(Weights, [1|convert_to_values(Inputs)], fun ml_math:sigmoid_deriv/1),
%%  Delta = calculate_delta(Backprop, Inputs, New_outputs,
%%    Output_value, Derv_value),
%%  io:format("(~w) New Sensitivities: ~w~n", [self(), New_outputs]),
%%  io:format("(~w) Calculated Sensitivity: ~w~n", [self(), Delta]),
%%
%%  % Adjust all the weights
%%  Weight_adjustments = lists:map(fun(Input) ->
%%    Learning_rate * Delta * Input
%%                                 end,
%%    convert_to_values(Inputs)),
%%  New_weights = lists:zipwith(fun(W, D) -> W + D end, tl(Weights), Weight_adjustments),
%%  io:format("(~w) Adjusted Weights: ~w~n", [self(), Weights]),
%%
%%  % propagate sensitivities and associated weights back to the previous layer
%%%%  lists:foreach(
%%%%    fun ({Weight, Input_PID}) ->
%%%%      neuron:learn(Input_PID, self(), Delta * Weight)
%%%%    end,
%%%%    lists:zip(New_weights, convert_to_keys(Inputs)),
%%%%  ),
%%%%  {noreply,  #state{weights = Weights, input_nodes = Inputs, output_nodes = Outputs, output_value = Output_value}};
%%  {noreply, #state{weights = New_weights, input_nodes = Inputs, output_nodes = New_outputs, activation = Output_value}};

handle_cast({connect_to_output, Output_node}, State) ->
  NewState = State#state{
    outputs = maps:put(Output_node, #output{}, State#state.outputs)
  },
  io:format("~w output connected to ~w: ~w~n", [self(), Output_node, maps:keys(NewState#state.outputs)]),
  {noreply, NewState};

handle_cast({connect_to_input, Input_node}, State) ->
  NewState = State#state{
    inputs = maps:put(Input_node, #input{}, State#state.inputs)
  },
  io:format("~w inputs connected to ~w: ~w~n", [self(), Input_node, maps:keys(NewState#state.inputs)]),
  {noreply, NewState};

handle_cast({pass, Input_value}, State) ->
  lists:foreach(
    fun(Output_Node) ->
      io:format("Stimulating ~w with ~w~n", [Output_Node, Input_value]),
      neuron:stimulate(Output_Node, self(), Input_value)
    end,
    maps:keys(State#state.outputs)),

  {noreply, State#state{activation =Input_value}};

handle_cast(_,_) ->
  erlang:error(not_implemented).

handle_call({set_weights, W}, _, State) ->
  New_State = lists:foldl(
    fun({Id, Weight}, S) ->
      Input = maps:get(Id, S#state.inputs),
      S#state{
        inputs = maps:update(
          Id,
          Input#input{weight = Weight},
          S#state.inputs
        )
      }
    end,
    State,
    W
  ),
  {reply, ok, New_State};

handle_call(get_output, _, State) ->
  {reply, State#state.activation, State};

handle_call(_, _, _) ->
  erlang:error(not_implemented).

handle_info(Other, State) ->
  io:format("Unexpected message: ~p~n",[Other]),
  io:format("State was: ~p~n",[State]),
  {noreply,State}.

terminate(Reason, State) ->
  io:format("Terminating for reason: ~p~n State was ~p~n",[Reason, State]),
  ok.

code_change(_OldVsn, State, _Extra) ->
  {ok, State}.

%%replace_input(Inputs, Input) ->
%%  lists:keyreplace(Input#input.node, 1, Inputs, Input).

%%% adds the propagating sensitivity to the Sensitivities Hash
%%add_delta(Deltas, Backprop) when Deltas =/= [] ->
%%  replace_input(Deltas, Backprop);
%%add_delta(Deltas, _Backprop) when Deltas =:= [] ->
%%  [].
%%
%%% Calculates the sensitivity of this particular node
%%calculate_delta(_, Inputs, Outputs, _, _)
%%  when Outputs =/= [], Inputs =:= [] -> % When the node is an input node:
%%  null;
%%calculate_delta(Delta, Inputs, Sensitivities, Output_value, Derv_value)
%%  when Sensitivities =:= [], Inputs =/= [] -> % When the node is an output node:
%%  {_, Training_value} = Delta,
%%  (Training_value - Output_value) * Derv_value;
%%calculate_delta(_, Inputs, Outputs, _, Derv_value)
%%  when Outputs =/= [], Inputs =/= [] -> % When the node is a hidden node:
%%  Derv_value * lists:foldl(fun(E, T) -> E + T end, 0, convert_to_values(Outputs)).
%%
%%convert_to_list(Inputs) ->
%%  lists:map(fun(Tup) ->
%%    {_, Val} = Tup,
%%    Val
%%            end,
%%    Inputs).
%%
%%convert_to_values(Tuple_list) ->
%%  lists:map(fun(Tup) ->
%%    {_, Val} = Tup,
%%    Val
%%            end,
%%    Tuple_list).
%%
%%convert_to_keys(Tuple_list) ->
%%  lists:map(fun(Tup) ->
%%    {Key, _} = Tup,
%%    Key
%%            end,
%%    Tuple_list).