-module(neuron).

%%-export([hyp/2, perceive/2, rand_weights/1]).
-compile(export_all).

-behavior(gen_server).

-record(state, {
  weights = [],
  inputs = [],
  output_Pids = []}).

%% API -----------------------------------------------------------------------------------------------------------------
start_link([W, I, OP]) -> gen_server:start_link(?MODULE, [W, I, OP], []).

stimulate(Pid, Inputs) -> gen_server:cast(Pid, {stimulate, Inputs}).

connect(Sender_PID, Receiver_PID) ->
  gen_server:cast(Sender_PID, {connect_to_output, Receiver_PID}),
  gen_server:cast(Receiver_PID, {connect_to_input, Sender_PID}).

pass(Pid, Input) -> gen_server:cast(Pid, {pass, Input}).

%% Debugging api
set_weights(Pid, W) -> gen_server:call(Pid, {set_weights, W }).

%%----------------------------------------------------------------------------------------------------------------------

hyp(W, A) -> ml_math:hyp(W, A, fun ml_math:sigmoid/1).

perceive(Inp, Weights) ->
    hyp([1|Inp], Weights).

init([W, I, OP]) ->
  {ok, #state{weights = W, inputs = I, output_Pids = OP }}.

handle_cast({stimulate, Input}, #state{weights=Weights, inputs=Inputs, output_Pids = Output_PIDs}) ->
  New_inputs = replace_input(Inputs, Input),
  Output = perceive(convert_to_list(New_inputs), Weights),

  if Output_PIDs =/= [] ->
    lists:foreach(fun(Output_PID) ->
      neuron:stimulate(Output_PID, {self(), Output})
                  end,
      Output_PIDs);
    Output_PIDs =:= [] ->
      io:format("~n~w outputs: ~w", [self(), Output])
  end,
  {noreply, #state{weights = Weights, inputs = New_inputs, output_Pids = Output_PIDs}};

handle_cast({connect_to_output, Receiver_PID}, State) ->
  Combined_output = [Receiver_PID | State#state.output_Pids],
  io:format("~w output connected to ~w: ~w~n", [self(), Receiver_PID, Combined_output]),
  {noreply, State#state{output_Pids = Combined_output}};

handle_cast({connect_to_input, Sender_PID}, #state{weights=Weights, inputs=Inputs, output_Pids = Output_PIDs}) ->
  Combined_input = [{Sender_PID, 0} | Inputs],
  io:format("~w inputs connected to ~w: ~w~n", [self(), Sender_PID, Combined_input]),
  {noreply, #state{weights = [rand:uniform() | Weights], inputs = Combined_input, output_Pids = Output_PIDs}};

handle_cast({pass, Input_value}, State) ->
  lists:foreach(fun(Output_PID) ->
    io:format("Stimulating ~w with ~w~n", [Output_PID, Input_value]),
    neuron:stimulate(Output_PID, {self(), Input_value})
                end,
    State#state.output_Pids),
  {noreply, State};

handle_cast(_,_) ->
  erlang:error(not_implemented).

handle_call({set_weights, W }, _, State) ->
  {reply, ok, State#state{weights = W}};

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