-module(neuron).

%%-export([hyp/2, perceive/2, rand_weights/1]).
-compile(export_all).

-behavior(gen_server).

-record(state, {
  weights = [],
  input_nodes = [],
  output_nodes = [],
  output_value = 0
  }).

%% API -----------------------------------------------------------------------------------------------------------------
start_link([W, I, OP]) -> gen_server:start_link(?MODULE, [W, I, OP], []).

stimulate(Node, Inputs) -> gen_server:cast(Node, {stimulate, Inputs}).

learn(NodeBehind, Node, Delta) -> gen_server:cast(NodeBehind, {learn, {Node, Delta}}).

connect(Sender_PID, Receiver_PID) ->
  gen_server:cast(Sender_PID, {connect_to_output, Receiver_PID}),
  gen_server:cast(Receiver_PID, {connect_to_input, Sender_PID}).

pass(Pid, Input) -> gen_server:cast(Pid, {pass, Input}).

%% Debugging api
set_weights(Pid, W) -> gen_server:call(Pid, {set_weights, W }).

get_output(Pid) -> gen_server:call(Pid, get_output).

%%----------------------------------------------------------------------------------------------------------------------

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).

perceive(Inp, Weights) ->
    hyp([1|Inp], Weights).

init([W, I, OP]) ->
  {ok, #state{weights = W, input_nodes = I, output_nodes = OP }}.

handle_cast({stimulate, Input}, #state{weights=Weights, input_nodes =Inputs, output_nodes = Output_nodes}) ->
  New_inputs = replace_input(Inputs, Input),
  Output = perceive(convert_to_list(New_inputs), Weights),

  if Output_nodes =/= [] ->
    lists:foreach(fun(Output_PID) ->
      neuron:stimulate(Output_PID, {self(), Output})
                  end,
      convert_to_keys(Output_nodes));
    Output_nodes =:= [] ->
      io:format("~n~w outputs: ~w", [self(), Output]),
      neuron:learn(self(), self(), 1)

  end,
  {noreply, #state{weights = Weights, input_nodes = New_inputs, output_nodes = Output_nodes, output_value = Output}};

handle_cast({learn, Backprop}, #state{weights=Weights, input_nodes = Inputs, output_nodes = Outputs}) ->

  Learning_rate = 0.5,

  % Calculate the correct sensitivities
  io:format("~p~n", [Outputs]),
  io:format("~p~n", [Backprop]),
  New_outputs = add_delta(Outputs, Backprop),
  Output_value = perceive(convert_to_list(Inputs), Weights),
  Derv_value = ml_math:hyp(Weights, [1|convert_to_values(Inputs)], fun ml_math:sigmoid_deriv/1),
  Delta = calculate_delta(Backprop, Inputs, New_outputs,
    Output_value, Derv_value),
  io:format("(~w) New Sensitivities: ~w~n", [self(), New_outputs]),
  io:format("(~w) Calculated Sensitivity: ~w~n", [self(), Delta]),

  % Adjust all the weights
  Weight_adjustments = lists:map(fun(Input) ->
    Learning_rate * Delta * Input
                                 end,
    convert_to_values(Inputs)),
  New_weights = lists:zipwith(fun(W, D) -> W + D end, tl(Weights), Weight_adjustments),
  io:format("(~w) Adjusted Weights: ~w~n", [self(), Weights]),

  % propagate sensitivities and associated weights back to the previous layer
%%  lists:foreach(
%%    fun ({Weight, Input_PID}) ->
%%      neuron:learn(Input_PID, self(), Delta * Weight)
%%    end,
%%    lists:zip(New_weights, convert_to_keys(Inputs)),
%%  ),
%%  {noreply,  #state{weights = Weights, input_nodes = Inputs, output_nodes = Outputs, output_value = Output_value}};
  {noreply, #state{weights = New_weights, input_nodes = Inputs, output_nodes = New_outputs, output_value = Output_value}};

handle_cast({connect_to_output, Receiver_PID}, State) ->
  Combined_output = [{Receiver_PID,1} | State#state.output_nodes],
  io:format("~w output connected to ~w: ~w~n", [self(), Receiver_PID, Combined_output]),
  {noreply, State#state{output_nodes = Combined_output}};

handle_cast({connect_to_input, Sender_PID}, #state{weights=Weights, input_nodes =Inputs, output_nodes = Output_PIDs}) ->
  Combined_input = [{Sender_PID, 0}| Inputs],
  io:format("~w inputs connected to ~w: ~w~n", [self(), Sender_PID, Combined_input]),
  {noreply, #state{weights = [rand:uniform() | Weights], input_nodes = Combined_input, output_nodes = Output_PIDs}};

handle_cast({pass, Input_value}, State) ->
  lists:foreach(fun(Output_PID) ->
    io:format("Stimulating ~w with ~w~n", [Output_PID, Input_value]),
    neuron:stimulate(Output_PID, {self(), Input_value})
                end,
    convert_to_keys(State#state.output_nodes)),
  {noreply, State#state{output_value =Input_value}};

handle_cast(_,_) ->
  erlang:error(not_implemented).

handle_call({set_weights, W }, _, State) ->
  {reply, ok, State#state{weights = W}};

handle_call(get_output, _, State) ->
  {reply, State#state.output_value, State};

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

replace_input(Inputs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

% adds the propagating sensitivity to the Sensitivities Hash
add_delta(Deltas, Backprop) when Deltas =/= [] ->
  replace_input(Deltas, Backprop);
add_delta(Deltas, _Backprop) when Deltas =:= [] ->
  [].

% Calculates the sensitivity of this particular node
calculate_delta(_, Inputs, Outputs, _, _)
  when Outputs =/= [], Inputs =:= [] -> % When the node is an input node:
  null;
calculate_delta(Delta, Inputs, Sensitivities, Output_value, Derv_value)
  when Sensitivities =:= [], Inputs =/= [] -> % When the node is an output node:
  {_, Training_value} = Delta,
  (Training_value - Output_value) * Derv_value;
calculate_delta(_, Inputs, Outputs, _, Derv_value)
  when Outputs =/= [], Inputs =/= [] -> % When the node is a hidden node:
  Derv_value * lists:foldl(fun(E, T) -> E + T end, 0, convert_to_values(Outputs)).

convert_to_list(Inputs) ->
  lists:map(fun(Tup) ->
    {_, Val} = Tup,
    Val
            end,
    Inputs).

convert_to_values(Tuple_list) ->
  lists:map(fun(Tup) ->
    {_, Val} = Tup,
    Val
            end,
    Tuple_list).

convert_to_keys(Tuple_list) ->
  lists:map(fun(Tup) ->
    {Key, _} = Tup,
    Key
            end,
    Tuple_list).