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
      FromPid ! {self(), Ref, ok},
      send_all(neuron:output_Ids(State), {setInput, Value}),
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
          ActivatedNeuron = neuron:calc_activation(N),
          send_all(neuron:output_Ids(ActivatedNeuron), {setInput, neuron:activation(ActivatedNeuron)}),
          loop(ActivatedNeuron);
        false ->
          loop(N)
      end;
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