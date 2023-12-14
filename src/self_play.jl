using Chess
include("board_class.jl")
include("mcts.jl")
include("model.jl")
include("test.jl")

function self_game(model::ChessNet, pos_dict::Dict{String, Tuple{SparseVector{Int, Int}, Int, Int}}, starting_position::String, arguments::Dict{String, Any})
	if starting_position == ""
		board = startboard()
	else
		board = fromfen(starting_position)
	end
	while !isterminal(board)
		move = tree_move(model, board, args)
		move = int_to_move(Int(only(move)))
        domove!(board, move)
	end

    result = 0
    if isdraw(board)
	    nothing
    elseif sidetomove(board) == WHITE
	    result = -1
    else
	    retuls = 1
    end
	return pos_dict
end
	


function self_play(model::ResNet, arguments::Dict{String, Any}, posisions_file::String)
	pos_dict = Dict{String, Tuple{SparseVector{Int, Int}, Int, Int}}()
	for game in arguments["num_games"]
		self_game(model, pos_dict)
end
