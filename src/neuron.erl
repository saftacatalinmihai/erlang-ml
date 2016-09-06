-module(neuron).

-compile(export_all).

%%-behavior(gen_server).

-record(state, {
  inputs,
  outputs,
  activation,
  inputs_received,
  bias,
  weights,
  lambda,
  batch_size,
  gradient_funs,
  gradients,
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

%% Callbacks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init([]) ->
  {ok, #state{
    inputs = [],
    inputs_received = 0,
    bias = rand:uniform() * 0.12 * 2 - 0.12,
    weights = [],
    activation_fn = fun ml_math:sigmoid/1,
    deriv_fn = fun ml_math:sigmoid_deriv/1,
    lambda = 1,
    outputs = [],
    batch_size = 10,
    gradient_funs = [],
    gradients = [],
    activation = 0
  }}.

handle_call({connect_to_output, Output_Node}, _, State) ->
  {reply, ok,State#state{
    outputs = [Output_Node | State#state.outputs]
  }};

handle_call({connect_to_input, Input_Node}, _, State) ->
  {reply, ok, State#state{
    inputs = [{Input_Node, 0} | State#state.inputs],
    weights = [rand:uniform() * 0.12 * 2 - 0.12 | State#state.weights]
  }};


handle_call(Request, From, State) ->
  erlang:error(not_implemented).

handle_cast({pass, Input}, State) ->
  lists:foreach(
    fun(Output_Node) ->
      neuron:forward(Output_Node, self(), Input)
    end,
    State#state.outputs),
  {noreply, State};

handle_cast({forward, FromPid, {Value, {expecting, ExpectedList}}}, State) ->
  NewInputs = lists:keyreplace(FromPid, 1, State#state.inputs, {FromPid, Value}),
  case all_inputs_received(State) of
    true ->
      case last_layer(State) of
        true ->
          Z = ml_math:dot_p([State#state.bias| State#state.weights], [1| input_values(NewInputs)]),
          A = (State#state.activation_fn)(Z),

          io:format("Predicted: ~p~n", [A]),
          Self = self(),
          case lists:keyfind(self(), 1, ExpectedList) of
            {Self, Expected} ->
              {noreply,
                learn(Expected, A, Z, State#state{
                  inputs = NewInputs,
                  inputs_received = 0,
                  activation = A
                })
              };
            false ->
              {noreply, State#state{
                inputs = NewInputs,
                inputs_received = 0,
                activation = A}
              }
          end;
        false ->
          {_, A, Gradient} = neuron_funs:forward(
            input_values(NewInputs),
            State#state.bias,
            State#state.weights,
            State#state.activation_fn,
            State#state.deriv_fn
          ),
          lists:foreach(fun(O) -> neuron:forward(O, self(), {A,{expecting, ExpectedList}}) end, State#state.outputs),
          {noreply, State#state{
            inputs = NewInputs,
            inputs_received = 0,
            activation = A,
            gradient_funs = [Gradient | State#state.gradient_funs]}
          }
      end;
    false ->
      {noreply, State#state{inputs = NewInputs, inputs_received = State#state.inputs_received + 1}}
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

last_layer(State) ->
  length(State#state.outputs) =:= 0.

learn(Expected, Activation, Z, State) ->
  BackProp = Expected - Activation,

  D = (State#state.deriv_fn)(Z),
  BiasGrad = BackProp * D,
  WeightsGrad =
    lists:map(
      fun(I) ->
        BackProp * D * I
      end,
      input_values(State#state.inputs)
    ),
  State#state{
    gradients = [{BiasGrad, WeightsGrad}]
  }.

input_values(Inputs) -> lists:map(fun({_, V}) -> V end, Inputs).