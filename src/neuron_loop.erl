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

% Loop
loop(State) ->
  receive
    {FromPid, Ref, getState} ->
      FromPid ! {self(), Ref, {ok, State}},
      loop(State);
    {FromPid, Ref, ping} ->
      FromPid ! {self(), Ref, {ok, pong}},
      loop(State);
    {FromPid, Ref, {addInput, Value}} ->
      FromPid ! {self(), Ref, ok},
      io:format("Adding input from ~p with value ~p~n", [FromPid, Value]),
      loop(
        neuron:calc_activation(
          neuron:add_input(State, FromPid, Value)
        )
      );
    {FromPid, Ref, {addOutputs, ToPids}} ->
      send_all(ToPids, {addInput, neuron:activation(State)}),
      FromPid ! {self(), Ref, ok},
      loop(neuron:add_outputs(State, ToPids));
    {FromPid, Ref, {setInput, Value}} ->
      FromPid ! {self(), Ref, ok},
      loop(neuron:set_input_value(State, FromPid, Value));
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

send_all(Pids, Msg) ->
  RefPids = lists:map(fun(P) -> {P, make_ref()} end, Pids),
  [P ! {self(), R, Msg} || {P, R} <- RefPids],
  wait_responses(RefPids).

wait_responses(RefPids) -> wait_responses(RefPids, []).
wait_responses([], Acc) -> Acc;
wait_responses(RefPids, Acc) ->
  receive
    {From, Ref, Msg} ->
%%      io:format("Received: ~p:~p:~p~n", [From, Ref, Msg]),
      wait_responses(lists:delete({From, Ref}, RefPids), [Msg | Acc])
  end.

test() ->
  {ok, I1} = ?MODULE:start(),
  {ok, I2} = ?MODULE:start(),
  {ok, H1} = ?MODULE:start(),
  {ok, H2} = ?MODULE:start(),
  {ok, O} = ?MODULE:start(),

  io:format("~p~n",[send_all([I1, I2, H1, H2, O], getState)]),
  addInput(I1, 1),
  addInput(I2, 1),

  addOutputs(I1, [H1, H2]),
  addOutputs(I2, [H1, H2]),
  addOutputs(H1, [O]),
  addOutputs(H2, [O]),

  io:format("~p~n",[send_all([I1, I2, H1, H2, O], getState)]),

  setInput(I1, 0),
  setInput(I2, 0),

  activate(I1),
  activate(I2),

  io:format("~p~n",[send_all([I1, I2, H1, H2, O], getState)]).
%%
%%all_inputs_received(State) ->
%%  State#state.inputs_received =:= length(State#state.inputs) - 1.
%%
%%all_backprops_received(State) ->
%%  State#state.backprops_received =:= length(State#state.outputs) - 1.
%%
%%last_layer(State) ->
%%  length(State#state.outputs) =:= 0.
