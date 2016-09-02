%%%-------------------------------------------------------------------
%%% @author mihai
%%% @copyright (C) 2016, <COMPANY>
%%% @doc
%%%
%%% @end
%%% Created : 02. Sep 2016 9:45 AM
%%%-------------------------------------------------------------------
-module(csv_parser).
-author("mihai").

%% API
-export([parse/1]).

parse(File) ->
  {ok, F} = file:open(File, [read, raw]),
  _Head = file:read_line(F),
  parse(F, file:read_line(F), []).

parse(F, eof, Acc) ->
  file:close(F),
  lists:reverse(Acc);

parse(F, {ok,Line}, Acc) ->
  parse(F, file:read_line(F), [parse_line(Line) | Acc]).

parse_line(Line) ->
  lists:map(
    fun(N) ->
      list_to_integer(
        lists:filter(
          fun(X) -> X =/= $\n end,
          N
        )
      )
    end,
    string:tokens(Line, ",")
  ).

%% csv_parser:parse("test.csv").
