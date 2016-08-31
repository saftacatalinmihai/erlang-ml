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
  outputs    = #{},                             %% map between output Node and record for backpropagation
  activation = 0                                %% last activation of this node
}).

%% API -----------------------------------------------------------------------------------------------------------------
start_link() -> gen_server:start_link(?MODULE, [], []).

stimulate(Node, FromNode, Activation) -> gen_server:cast(Node, {stimulate, FromNode, Activation}).

learn(BackNode, From_Node, Backprop) ->
  gen_server:cast(BackNode, {learn, From_Node, Backprop}).

learn(Node, Expected) ->
  gen_server:cast(Node, {learn, Expected}).

connect(Input_node, Output_Node) ->
  gen_server:cast(Input_node, {connect_to_output, Output_Node}),
  gen_server:cast(Output_Node, {connect_to_input, Input_node}).

pass(Pid, Input) -> gen_server:cast(Pid, {pass, Input}).

%% Debugging api
set_weights(Pid, W) -> gen_server:call(Pid, {set_weights, W }).

get_output(Pid) -> gen_server:call(Pid, get_output).

%%----------------------------------------------------------------------------------------------------------------------

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).
hyp_deriv(W, X) -> ml_math:hyp(W, X, fun ml_math:sigmoid_deriv/1).

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

%%  case maps:size(State#state.outputs) of
%%    0 ->
%%      io:format("~n~w Activation: ~w", [self(), Neuron_Activation]),
%%      lists:foreach(
%%          fun(InputNode) ->
%%            neuron:learn(InputNode, self(), Neuron_Activation - Expected)
%%          end,
%%        maps:keys(State#state.outputs)
%%      );
%%    _ ->
      lists:foreach(
        fun(Output_Node) ->
          neuron:stimulate(Output_Node, self(), Neuron_Activation)
        end,
        maps:keys(State#state.outputs)
      ),
%%  end,
  {noreply, State#state{inputs = New_Inputs, activation = Neuron_Activation}};

handle_cast({learn, Expected}, State) ->
  Deriv = hyp_deriv(
    lists:map(fun(I) -> I#input.activation end, maps:values(State#state.inputs)),
    lists:map(fun(I) -> I#input.weight     end, maps:values(State#state.inputs))
  ),
  Delta = (Expected - State#state.activation) * Deriv,
  io:format("--Error: ~p~n", [(Expected - State#state.activation)*(Expected - State#state.activation)]),
  New_Inputs = maps:map(
    fun (_, Input) ->
      Input#input{
        weight = Input#input.weight + 0.1 * Delta * Input#input.activation   %% TODO: remove hardcoded 0.5 learning rate
      }
    end,
    State#state.inputs
  ),

  lists:foreach(
    fun (Input_Node) ->
      Node = maps:get(Input_Node, New_Inputs),
      neuron:learn(
        Input_Node,
        self(),
        Node#input.weight * Delta
      )
    end,
    lists:filter(fun(K) -> K =/= 1 end, maps:keys(New_Inputs))              %% remove the bias input node
  ),
  {noreply, State#state{inputs = New_Inputs}};

handle_cast({learn, FromNode, BackProp}, State) ->
  io:format("Learing: ~p~n", [[FromNode, BackProp]]),

  New_outputs = maps:update(
    FromNode,
    #output{backprop = BackProp},
    State#state.outputs
  ),

  Deriv = hyp_deriv(
    lists:map(fun(I) -> I#input.activation end, maps:values(State#state.inputs)),
    lists:map(fun(I) -> I#input.weight     end, maps:values(State#state.inputs))
  ),
  Delta = Deriv * lists:sum(
    lists:map(fun ({_,Output}) -> Output#output.backprop end, maps:to_list(State#state.outputs))
  ),
  New_Inputs = maps:map(
    fun (_, Input) ->
      Input#input{
        weight = Input#input.weight + 0.1 * Delta * Input#input.activation   %% TODO: remove hardcoded 0.5 learning rate
      }
    end,
    State#state.inputs
  ),

  lists:foreach(
    fun (Input_Node) ->
      Node = maps:get(Input_Node, New_Inputs),
      neuron:learn(
        Input_Node,
        self(),
        Node#input.weight * Delta
      )
    end,
    lists:filter(fun(K) -> K =/= 1 end, maps:keys(New_Inputs))              %% remove the bias input node
  ),
  {noreply, State#state{inputs = New_Inputs, outputs = New_outputs}};

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