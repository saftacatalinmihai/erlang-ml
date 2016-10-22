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

-record(state, {
  neuron = neuron:new(),
  backprop_queue = []
}).

%% API
start() ->
  Pid = spawn(?MODULE, loop, [#state{}]),
  {ok, Pid}.

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
    {Pid, R, {ok, State}} -> neuron:activation(State#state.neuron)
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
      loop(State#state{neuron = neuron:set_input_bias(State#state.neuron, B)});

    {FromPid, Ref, {setWeight, InputId, W}} ->
      FromPid ! {self(), Ref, ok},
      loop(State#state{neuron = neuron:set_input_weight(State#state.neuron, InputId, W)});

    {FromPid, Ref, {pass, Value}} ->
      FromPid ! {self(), Ref, ok},
      send_all(neuron:output_Ids(State#state.neuron), {setInput, Value}),
      loop(State);

    {FromPid, Ref, {addInput, Value}} ->
      FromPid ! {self(), Ref, ok},
      loop(State#state{neuron = neuron:add_input(State#state.neuron, FromPid, Value)});

    {FromPid, Ref, {addOutputs, ToPids}} ->
      FromPid ! {self(), Ref, ok},
      send_all(ToPids, {addInput, neuron:activation(State#state.neuron)}),
      loop(State#state{neuron = neuron:add_outputs(State#state.neuron, ToPids)});

    {FromPid, Ref, {setInput, Value}} ->
      N = neuron:set_input_value(State#state.neuron, FromPid, Value),
      io:format("Setting input from ~p with value ~p~n", [FromPid, Value]),
      case neuron:all_inputs_set(N) of
        true ->
          ActivatedNeuron = neuron:calc_activation(N),

          Delta = case send_all(neuron:output_Ids(ActivatedNeuron), {setInput, neuron:activation(ActivatedNeuron)}) of
            [] -> 1 - neuron:activation(ActivatedNeuron);
            Deltas -> lists:sum(Deltas)
          end,
          io:format("BackProp: ~p~n", [Delta]),
          FromPid ! {self(), Ref, neuron:get_input_weight(ActivatedNeuron, FromPid) * Delta},
          lists:foreach(
            fun ({InputPid, R}) ->
              InputPid ! {self(), R, neuron:get_input_weight(ActivatedNeuron, InputPid) * Delta}
            end,
            State#state.backprop_queue
          ),
          loop(State#state{neuron = neuron:update_weights(ActivatedNeuron, Delta), backprop_queue = []});
        false ->
          % TODO: save Ref to be used to back-propagate delta
          loop(State#state{neuron = N, backprop_queue = [{FromPid, Ref} | State#state.backprop_queue]})
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
    {P,R, Msg} -> wait_responses(lists:delete({P, R}, RefPids), [Msg | Acc])
%%    _Other ->
%%      self() ! Other,
%%      wait_responses(RefPids, Acc)
  end.