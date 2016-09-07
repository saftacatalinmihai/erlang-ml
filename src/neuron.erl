-module(neuron).

-compile(export_all).

%%-behavior(gen_server).

-record(state, {
  inputs,
  outputs,
  activation,
  inputs_received,
  backprops_received,
  latest_backprop,
  bias,
  weights,
  lambda,
  batch_size,
  gradient_funs,
  activation_fn,
  deriv_fn
}).

%% API %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start() ->
  gen_server:start_link(?MODULE, [], []).
start_link() ->
  gen_server:start_link(?MODULE, [], []).

connect(Input_Node, Output_Node) ->
  gen_server:call(Input_Node, {connect_to_output, Output_Node}),
  gen_server:call(Output_Node, {connect_to_input, Input_Node}).

pass(Pid, Input)             -> gen_server:cast(Pid, {pass, Input}).
forward(Pid, FromPid, Input) -> gen_server:cast(Pid, {forward, FromPid, Input}).
backprop(Pid, FromPid, BackProp) ->gen_server:cast(Pid, {backprop, FromPid, BackProp}).

%% Debug API
set_weights(Pid, Bias, Weights) -> gen_server:call(Pid, {set_weights, Bias, Weights}).

%% Callbacks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init([]) ->
  {ok, #state{
    inputs = [],
    inputs_received = 0,
    backprops_received = 0,
    bias = rand:uniform() * 0.12 * 2 - 0.12,
    weights = [],
    activation_fn = fun ml_math:sigmoid/1,
    deriv_fn = fun ml_math:sigmoid_deriv/1,
    lambda = 1,
    outputs = [],
    batch_size = 10,
    gradient_funs = [],
    activation = 0,
    latest_backprop = 0
  }}.

handle_call({connect_to_output, Output_Node}, _, State) ->
  {reply, ok,State#state{
    outputs = [{Output_Node, 0}| State#state.outputs]
  }};

handle_call({connect_to_input, Input_Node}, _, State) ->
  {reply, ok, State#state{
    inputs = [{Input_Node, 0} | State#state.inputs],
    weights = [rand:uniform() * 0.12 * 2 - 0.12 | State#state.weights]
  }};

handle_call({set_weights, Bias, Weights}, _, State) ->
  {reply, ok, State#state{
    bias = Bias,
    weights = Weights
  }};

handle_call(Request, From, State) ->
  erlang:error(not_implemented).

handle_cast({pass, Input}, State) ->
  lists:foreach(
    fun(Output_Node) ->
      neuron:forward(Output_Node, self(), Input)
    end,
    output_nodes(State#state.outputs)),
  {noreply, State};

handle_cast({forward, FromPid, {Value, {expecting, ExpectedList}}}, State) ->
  NewInputs = lists:keyreplace(FromPid, 1, State#state.inputs, {FromPid, Value}),
  case all_inputs_received(State#state{inputs = NewInputs}) of
    true ->
      case last_layer(State) of
        true ->
          on_forward_last_layer(State#state{inputs = NewInputs}, ExpectedList);

        false ->
          on_forward_hidden_layer(State#state{inputs = NewInputs}, ExpectedList)

      end;
    false ->
      {noreply, State#state{inputs = NewInputs, inputs_received = State#state.inputs_received + 1}}
  end;

handle_cast({backprop, FromNode, BackProp}, State) ->
  NewOutputs = lists:keyreplace(FromNode, 1, State#state.outputs, {FromNode, BackProp}),
  case all_backprops_received(State#state{outputs = NewOutputs}) of
    true ->
      BackPropSum = lists:sum(output_backprops(NewOutputs)),
      io:format("Received Backprops: ~p ~p~n", [self(), output_backprops(NewOutputs)]),
      backprop_(State, BackPropSum),

      Gradient = neuron_funs:gradient(BackPropSum, State#state.bias, State#state.weights, input_values(State#state.inputs), State#state.deriv_fn),

      {NewBias, NewWeights} = neuron_funs:backprop(State#state.bias, State#state.weights, State#state.lambda, [Gradient]),
      {noreply, State#state{bias = NewBias, weights = NewWeights, outputs = NewOutputs, gradient_funs = [], backprops_received = 0, latest_backprop = BackPropSum}};
    false ->
      {noreply, State#state{outputs = NewOutputs, backprops_received = State#state.backprops_received + 1}}
  end;

handle_cast(Request, State) ->
  io:format("Not implemented Request: ~p~n State was ~p~n",[Request, State]),
  erlang:error(not_implemented).

handle_info(_, State) ->
  io:format("State: ~p~n", [State]),
  {noreply,State}.

terminate(Reason, State) ->
  io:format("Terminating for reason: ~p~n State was ~p~n",[Reason, State]),
  ok.

code_change(OldVsn, State, Extra) ->
  erlang:error(not_implemented).

%%% Custom functions

all_inputs_received(State) ->
  State#state.inputs_received =:= length(State#state.inputs) - 1.

all_backprops_received(State) ->
  State#state.backprops_received =:= length(State#state.outputs) - 1.

last_layer(State) ->
  length(State#state.outputs) =:= 0.

on_forward_last_layer(State, ExpectedList) ->
  Z = ml_math:dot_p([State#state.bias| State#state.weights], [1| input_values(State#state.inputs)]),
  A = (State#state.activation_fn)(Z),

  io:format("Predicted: ~p~n", [A]),
  Self = self(),
  case lists:keyfind(self(), 1, ExpectedList) of
    {Self, Expected} ->
      {noreply,
        learn(Expected, A, Z, State#state{
          inputs_received = 0,
          activation = A
        })
      };
    false ->
      {noreply, State#state{
        inputs_received = 0,
        activation = A}
      }
  end.

on_forward_hidden_layer(State, ExpectedList) ->
  {_, A, Gradient_fn} = neuron_funs:forward(
    input_values(State#state.inputs),
    State#state.bias,
    State#state.weights,
    State#state.activation_fn,
    State#state.deriv_fn
  ),
  lists:foreach(fun(O) -> neuron:forward(O, self(), {A,{expecting, ExpectedList}}) end, output_nodes(State#state.outputs)),

  Gradient = Gradient_fn(State#state.latest_backprop),
  {NewBias, NewWeights} = neuron_funs:backprop(State#state.bias, State#state.weights, State#state.lambda, [Gradient]),

  backprop_(State#state{bias = NewBias, weights = NewWeights}, State#state.latest_backprop),

  {noreply, State#state{
    inputs_received = 0,
    activation = A,
    bias = NewBias,
    weights = NewWeights,
    gradient_funs = [Gradient_fn | State#state.gradient_funs]}
  }.

backprop_(State, BackProp) ->
  lists:foreach(
    fun ({W,N}) ->
      neuron:backprop(N,self(),W*BackProp)
    end,
    lists:zip(State#state.weights, input_nodes(State#state.inputs))
  ).

learn(Expected, Activation, Z, State) ->
  BackProp = Expected - Activation,

  backprop_(State, BackProp),

  D = (State#state.deriv_fn)(Z),
  BiasGrad = BackProp * D,
  WeightsGrad =
    lists:map(
      fun(I) ->
        BackProp * D * I
      end,
      input_values(State#state.inputs)
    ),
  {NewBias, NewWeights} = neuron_funs:backprop(State#state.bias, State#state.weights, State#state.lambda, [{BiasGrad, WeightsGrad}]),

  State#state{
    bias = NewBias,
    weights = NewWeights
  }.

input_values(Inputs) -> lists:map(fun({_, V}) -> V end, Inputs).
input_nodes (Inputs) -> lists:map(fun({Node, _}) -> Node end, Inputs).

output_nodes(Outputs) -> lists:map(fun({Node, _}) -> Node end, Outputs).
output_backprops(Outputs) -> lists:map(fun({_, BP}) -> BP end, Outputs).

