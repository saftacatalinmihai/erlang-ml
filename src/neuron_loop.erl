%%%-------------------------------------------------------------------
%%% @author mihai
%%% @copyright (C) 2016, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 15. Sep 2016 5:09 PM
%%%-------------------------------------------------------------------
-module(neuron_loop).
-author("mihai").

-compile(export_all).
%% API

start() ->
  Pid = spawn(?MODULE, loop, [neuron:new()]),
  {ok, Pid}.

addInput(Pid, V) ->
  R = make_ref(),
  Pid ! {self(), R, {addInput, V}},
  receive
    {Pid, R, M} -> M
  end.

addOutputs(Pid, ToPids) ->
  R = make_ref(),
  Pid ! {self(), R, {addOutputs, ToPids}},
  receive
    {Pid, R, M} -> M
  end.

pass(Pid, Value) ->
  R = make_ref(),
  Pid ! {self(), R, {pass, Value}},
  receive
    {Pid, R, M} -> M
  end.

setInput(Pid, Value) ->
  R = make_ref(),
  Pid ! {self(), R, {setInput, Value}},
  receive
    {Pid, R, M} -> M
  end.

activate(Pid) ->
  R = make_ref(),
  Pid ! {self(), R, activate},
  receive
    {Pid, R, M} -> M
  end.

% DEBUG
getActivation(Pid) ->
  R = make_ref(),
  Pid ! {self(), R, getState},
  receive
    {Pid, R, {ok, N}} -> neuron:activation(N)
  end.

setWeight(Pid, InputId, Weight) ->
  R = make_ref(),
  Pid ! {self(), R, {setWeight, InputId, Weight}},
  receive
    {Pid, R, M} -> M
  end.

setBias(Pid, B) ->
  R = make_ref(),
  Pid ! {self(), R, {setBias, B}},
  receive
    {Pid, R, M} -> M
  end.


% Loop
loop(State) ->
  receive
    {FromPid, Ref, getState} ->
      FromPid ! {self(), Ref, {ok, State}},
      loop(State);

    {FromPid, Ref, ping} ->
      FromPid ! {self(), Ref, {ok, pong}},
      loop(State);

    {FromPid, Ref, {setBias, B}} ->
      FromPid ! {self(), Ref, ok},
      loop(neuron:set_input_bias(State, B));

    {FromPid, Ref, {setWeight, InputId, W}} ->
      FromPid ! {self(), Ref, ok},
      loop(neuron:set_input_weight(State, InputId, W));

    {FromPid, Ref, {pass, Value}} ->
      send_all(neuron:output_Ids(State), {setInput, Value}),
      FromPid ! {self(), Ref, ok},
      loop(State);

    {FromPid, Ref, {addInput, Value}} ->
      FromPid ! {self(), Ref, ok},
      loop(neuron:add_input(State, FromPid, Value));

    {FromPid, Ref, {addOutputs, ToPids}} ->
      FromPid ! {self(), Ref, ok},
      send_all(ToPids, {addInput, neuron:activation(State)}),
      loop(neuron:add_outputs(State, ToPids));

    {FromPid, Ref, {setInput, Value}} ->
      FromPid ! {self(), Ref, ok},
      N = neuron:set_input_value(State, FromPid, Value),
      io:format("Setting input from ~p with value ~p~n", [FromPid, Value]),
      case neuron:all_inputs_set(N) of
        true ->
          N2 = neuron:calc_activation(N),
          A = neuron:activation(N2),
          send_all(neuron:output_Ids(N2), {setInput, A}),
          loop(N2);
        false ->
          loop(N)
      end;

    {FromPid, Ref, activate} ->
      N = neuron:calc_activation(State),
      A = neuron:activation(N),
      send_all(neuron:output_Ids(State), {setInput, A}),
%%      send_all(neuron:output_Ids(State), activate),
      FromPid ! {self(), Ref, ok},
      loop(N);
%%    {FromPid, {forward, {Value, ExpectedList}}} ->
%%      NewInputs = lists:keyreplace(FromPid, 1, State#state.inputs, {FromPid, Value}),
%%      case all_inputs_received(State#state{inputs = NewInputs}) of
%%        true ->
%%          case last_layer(State) of
%%            true ->
%%              on_forward_last_layer(State#state{inputs = NewInputs}, ExpectedList);
%%
%%            false ->
%%              on_forward_hidden_layer(State#state{inputs = NewInputs}, ExpectedList)
%%
%%          end;
%%        false ->
%%          {noreply, State#state{inputs = NewInputs, inputs_received = State#state.inputs_received + 1}}
%%      end;
%%      io:format("Received msg: ~p~n", [X]),
%%      From ! {self(), {backprop, X}},
%%      loop(State)
    Other -> io:format("Unhanled message: ~p~n", [Other])
  end.


%% Guaraties respnses are in the order of the Pids
send_all(Pids, Msg) ->
  RefPids = lists:map(fun(P) -> {P, make_ref()} end, Pids),
  [P ! {self(), R, Msg} || {P, R} <- RefPids],
  wait_responses(RefPids).

wait_responses(RefPids) -> wait_responses(RefPids, []).
wait_responses([], Acc) -> lists:reverse(Acc);
wait_responses([{P,R}|RefPids], Acc) ->
  receive
    {P,R, Msg} -> wait_responses(lists:delete({P, R}, RefPids), [Msg | Acc]);
    _ -> wait_responses(RefPids, Acc)
  end.

%%all_inputs_received(State) ->
%%  State#state.inputs_received =:= length(State#state.inputs) - 1.
%%
%%all_backprops_received(State) ->
%%  State#state.backprops_received =:= length(State#state.outputs) - 1.
%%
%%last_layer(State) ->
%%  length(State#state.outputs) =:= 0.
