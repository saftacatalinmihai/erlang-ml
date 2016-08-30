-module(neuron).

%%-export([hyp/2, perceive/2, rand_weights/1]).
-compile(export_all).

-behavior(gen_server).

-record(state, {
  weights = [],
  input_nodes = [],
  output_nodes = [],
  output = 0
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
      Output_nodes);
    Output_nodes =:= [] ->
      io:format("~n~w outputs: ~w", [self(), Output])
%%      neuron:learn(self(), self(), 1)

  end,
  {noreply, #state{weights = Weights, input_nodes = New_inputs, output_nodes = Output_nodes, output = Output}};

handle_cast({learn, _Delta}, State) ->
  {noreply, State};

handle_cast({connect_to_output, Receiver_PID}, State) ->
  Combined_output = [Receiver_PID | State#state.output_nodes],
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
    State#state.output_nodes),
  {noreply, State#state{output=Input_value}};

handle_cast(_,_) ->
  erlang:error(not_implemented).

handle_call({set_weights, W }, _, State) ->
  {reply, ok, State#state{weights = W}};

handle_call(get_output, _, State) ->
  {reply, State#state.output, State};

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