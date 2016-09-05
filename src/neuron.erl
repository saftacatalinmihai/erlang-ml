-module(neuron).

%%-export([hyp/2, perceive/2, rand_weights/1]).
-compile(export_all).

-behavior(gen_server).

-record( input, {
  weight = rand:uniform() * 0.12 * 2 - 0.12, %% the weight associated to the input stimulus
  value  = 0,                                %% the last activation of input node
  hist   = #{}                               %% map between training example ID and input value
}).

-record( output, {
  backprop = 0                               %% backprop = weight * delta
}).

-record( state, {
  inputs     = #{1 => #input{value = 1}},    %% map between input Node and record. Starts with the bias term id = 1
  outputs    = #{},                          %% map between output Node and record for backpropagation
  inputs_received  = 1,
  outputs_received = 0,
  activation = 0,                            %% last activation of this node
  callbackOnOut = fun (_) -> null end,
  backprop_hist = [],                        %% map between training example ID and backprop value
  batch_size = 10
}).

%% API -----------------------------------------------------------------------------------------------------------------
start_link(CallbackOnOutput) ->
  gen_server:start_link(?MODULE, [CallbackOnOutput], []).
start_link() ->
  gen_server:start_link(?MODULE, [], []).

start_multi(Num) ->
  L = [neuron:start_link() || _ <- lists:seq(1, Num)],
  lists:map( fun ({ok, N}) -> N end, L).

start_multi(Num, CallbackOnOutput) ->
  L = [neuron:start_link(CallbackOnOutput) || _ <- lists:seq(1, Num)],
  lists:map( fun ({ok, N}) -> N end, L).

full_connect(In, Out) ->
  lists:foreach(
    fun(I) ->
      lists:foreach(
        fun(O) ->
          neuron:connect(I, O)
        end,
        Out
      )
    end,
    In
  ).

pass(Pid, {predict, Value}) -> gen_server:cast(Pid, {pass, {predict, Value}});
pass(Pid, {train, Example_ref, Value}) -> gen_server:cast(Pid, {pass, {train, Example_ref, Value}});
pass(Pid, Input           ) -> gen_server:cast(Pid, {pass, {predict, Input}}).

stimulate(Node, FromNode, Input) -> gen_server:cast(Node, {stimulate, FromNode, Input}).

learn(BackNode, From_Node, {Example_ref, Backprop}) ->
  gen_server:cast(BackNode, {learn, From_Node, {Example_ref, Backprop}}).

learn(Node, {Example_ref, Expected}) ->
  gen_server:cast(Node, {learn, {Example_ref, Expected}}).

connect(Input_node, Output_Node) ->
  gen_server:cast(Input_node, {connect_to_output, Output_Node}),
  gen_server:cast(Output_Node, {connect_to_input, Input_node}).


%% Debugging api
set_weights(Pid, W) -> gen_server:call(Pid, {set_weights, W }).
get_state(Pid)      -> gen_server:call(Pid, get_state).

get_output(Pid) -> gen_server:call(Pid, get_output).

%%----------------------------------------------------------------------------------------------------------------------

init([CallbackOnOutput]) ->
  {ok, #state{callbackOnOut = CallbackOnOutput}};
init([]) ->
  {ok, #state{}}.

handle_cast({stimulate, FromNode, Input}, State) ->
  {noreply, stimulate_(FromNode, Input, State)};

handle_cast({pass, Input_value}, State) ->
  {noreply, pass_(Input_value, State)};

handle_cast({connect_to_output, Output_node}, State) ->
  NewState = State#state{
    outputs = maps:put(Output_node, #output{}, State#state.outputs)
  },
%%  io:format("~w output connected to ~w: ~w~n", [self(), Output_node, maps:keys(NewState#state.outputs)]),
  {noreply, NewState};

handle_cast({connect_to_input, Input_node}, State) ->
  NewState = State#state{
    inputs = maps:put(Input_node, #input{}, State#state.inputs)
  },
%%  io:format("~w inputs connected to ~w: ~w~n", [self(), Input_node, maps:keys(NewState#state.inputs)]),
  {noreply, NewState};



handle_cast({learn, {Example_ref, Expected}}, State) ->
  Delta = (Expected - State#state.activation),
%%  io:format("--Error: ~p~n", [Delta*Delta]),

  BackpropHist = [{Example_ref, Delta} | State#state.backprop_hist],

  {NI, BPH} = case length(BackpropHist) =:= State#state.batch_size of
    true ->
      New_Inputs = maps:map(
        fun (_, Input) ->
          Sum = lists:foldl(
            fun (Ex_ref, Acc) ->
              #{value := Value, z := Z} = maps:get(Ex_ref, Input#input.hist),
              {Ex_ref, BackProp} = lists:keyfind(Ex_ref, 1, BackpropHist),
              Acc + BackProp * ml_math:sigmoid_deriv(Z) * Value
            end,
            0,
            maps:keys(Input#input.hist)
          ),

          Input#input{
            weight = Input#input.weight + 0.5 * Sum   %% TODO: remove hardcoded 0.5 learning rate
          }
        end,
        State#state.inputs
      ),
      backprop(State, Delta, New_Inputs, Example_ref),
      {New_Inputs, []};
    _ ->
      backprop(State, Delta, State#state.inputs, Example_ref),
      {State#state.inputs, BackpropHist}
  end,

  {noreply, State#state{inputs = NI, backprop_hist = BPH}};

handle_cast({learn, FromNode, {Example_ref, BackProp}}, State) ->
%%  io:format("Learing: ~p~n", [[FromNode, BackProp]]),

  New_outputs = maps:update(
    FromNode,
    #output{backprop = BackProp},
    State#state.outputs
  ),

  {New_outputs_received, NI, BPH} = case State#state.outputs_received =:= maps:size(State#state.outputs) - 1 of
                                 true ->

                                   Delta = lists:sum(
                                     lists:map(fun(O) -> O#output.backprop end, maps:values(New_outputs))
                                   ),

                                   BackpropHist = [{Example_ref, Delta}| State#state.backprop_hist],

                                   case length(BackpropHist) =:= State#state.batch_size of
                                      true ->
                                        New_Inputs = calculate_new_weights(State#state{backprop_hist = BackpropHist}),
                                        backprop(State, Delta, New_Inputs, Example_ref),
                                        {0, New_Inputs, []};
                                      _ ->
                                        backprop(State, Delta, State#state.inputs, Example_ref),
                                        {0, State#state.inputs, BackpropHist}
                                   end;

                                 false ->
                                   {State#state.outputs_received + 1, State#state.inputs, State#state.backprop_hist}
                               end,

  {noreply, State#state{inputs = NI, outputs = New_outputs, outputs_received = New_outputs_received, backprop_hist = BPH}};

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

handle_call(get_state, _, State) ->
  io:format("Getting state ~n"),
  io:format("~p~n", [State]),
  {reply, State, State};

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

sig_prime(State) ->
  Inputs = maps:to_list(State#state.inputs),
  A = lists:map(fun({_, I}) -> I#input.value end, Inputs),
  W = lists:map(fun({_, I}) -> I#input.weight     end, Inputs),
  hyp_deriv(A, W).

backprop(State, Delta, _New_inputs, Example_ref) ->
  lists:foreach(
    fun (Input_Node) ->
      Node = maps:get(Input_Node, State#state.inputs),
      neuron:learn(
        Input_Node,
        self(),
        {Example_ref, Node#input.weight * Delta}
      )
    end,
    lists:filter(fun(K) -> K =/= 1 end, maps:keys(State#state.inputs))              %% remove the bias input node
  ).

calculate_new_weights(State)->
  io:format("Computing new weights~n"),
  io:format("State: ~p~n", [State]),
  maps:map(
    fun (_, Input) ->
      Sum = lists:foldl(
        fun (Ex_ref, Acc) ->
          #{value := Value, z := Z} = maps:get(Ex_ref, Input#input.hist),
          {Ex_ref, BackProp} = lists:keyfind(Ex_ref, 1, State#state.backprop_hist),
          Acc + BackProp * ml_math:sigmoid_deriv(Z) * Value
        end,
        0,
        maps:keys(Input#input.hist)
      ),

      Input#input{
        weight = Input#input.weight + 0.5 * Sum   %% TODO: remove hardcoded 0.5 learning rate
      }
    end,
    State#state.inputs
  ).

stimulate_(FromNode, {predict, Value}, State) ->
  #{FromNode := InputNode} = State#state.inputs,

  New_Inputs = maps:update(
    FromNode,
    InputNode#input{value = Value},
    State#state.inputs
  ),

  {New_inputs_received, NA} = case State#state.inputs_received =:= maps:size(State#state.inputs) - 1 of
                                true ->
                                  InputList = maps:values(New_Inputs),
                                  Neuron_Activation = perceive(
                                    lists:map(fun(I) -> I#input.value  end, InputList),
                                    lists:map(fun(I) -> I#input.weight end, InputList)
                                  ),
                                  (State#state.callbackOnOut)([self(), Neuron_Activation]),

                                  lists:foreach(
                                    fun(Output_Node) ->
                                      neuron:stimulate(Output_Node, self(), {predict, Neuron_Activation})
                                    end,
                                    maps:keys(State#state.outputs)
                                  ),
                                  {1, Neuron_Activation};
                                _ ->
                                  {State#state.inputs_received + 1, State#state.activation}
                              end,

  State#state{
    inputs = New_Inputs,
    inputs_received = New_inputs_received,
    activation = NA
  };

stimulate_(FromNode, {train, Example_ref, Value}, State) ->
  #{FromNode := InputNode} = State#state.inputs,

  Hist = maps:put(
    Example_ref,
    #{value => Value}, InputNode#input.hist),

  New_Inputs = maps:update(
    FromNode,
    InputNode#input{value = Value, hist = Hist},
    State#state.inputs
  ),

  {New_inputs_received, NA, NI} = case State#state.inputs_received =:= maps:size(State#state.inputs) - 1 of
                                true ->
                                  InputList = maps:values(New_Inputs),
                                  Z = ml_math:dot_p(
                                    lists:map(fun(I) -> I#input.value  end, InputList),
                                    lists:map(fun(I) -> I#input.weight end, InputList)
                                  ),
                                  Neuron_Activation = ml_math:sigmoid(Z),

                                  NewHist = maps:update(
                                    Example_ref,
                                    maps:put(z, Z , maps:get(Example_ref, Hist)), Hist),
                                  NewHist2 = maps:update(
                                    Example_ref,
                                    maps:put(a, Neuron_Activation, maps:get(Example_ref, NewHist)), NewHist),

                                  New_Inputs2 = maps:update(
                                    FromNode,
                                    InputNode#input{
                                      hist = NewHist2
                                    },
                                    New_Inputs
                                  ),
%%                                  io:format("new inputs: ~p~n", [New_Inputs2]),

                                  (State#state.callbackOnOut)([self(), Neuron_Activation]),

                                  lists:foreach(
                                    fun(Output_Node) ->
                                      neuron:stimulate(Output_Node, self(), {train, Example_ref, Neuron_Activation})
                                    end,
                                    maps:keys(State#state.outputs)
                                  ),
                                  {1, Neuron_Activation, New_Inputs2};
                                _ ->
                                  {State#state.inputs_received + 1, State#state.activation, New_Inputs}
                              end,

  State#state{
    inputs = NI,
    inputs_received = New_inputs_received,
    activation = NA
  }.

pass_(Input_value, State) ->
  lists:foreach(
    fun(Output_Node) ->
      neuron:stimulate(Output_Node, self(), Input_value)
    end,
    maps:keys(State#state.outputs)),
  State#state{activation = Input_value}.

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).
hyp_deriv(W, X) -> ml_math:hyp(W, X, fun ml_math:sigmoid_deriv/1).

perceive(Inp, Weights) ->
  hyp(Inp, Weights).

