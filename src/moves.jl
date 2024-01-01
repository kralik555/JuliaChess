using Chess
using Flux
using JLD2
include("board_class.jl")
include("model.jl")
include("mcts.jl")

function model_move(model::ChessNet, board::Board)
    # Convert the board to a tensor representation suitable for your model
	tensor = board_to_tensor(board)
	moves, value = model.model(tensor)
	valid_moves = get_valid_moves(board)                           
    moves = vec(moves)                                                
    moves = moves .* valid_moves                                      
    moves = moves ./ sum(moves)                                      

    # Choose the move with the highest probability
    best_move_index = argmax(moves[:])
	return best_move_index
end

function tree_move(model::ChessNet, game::SimpleGame, args::Dict{String, Float64})
	tree = MCTS(game, args, model)
	move_probs = search(tree)
	move = argmax(move_probs)
	return move
end

function tree_move_with_distro(model::ChessNet, game::SimpleGame, args::Dict{String, Float64})
    tree = MCTS(game, args, model)
    move_probs = search(tree)
    move = argmax(move_probs)
    return (move_probs, move)
end

function model_move_with_distros(model::ChessNet, board::Board)
    probs, _ = model.model(board_to_tensor(game.board))
    valid_moves = get_valid_moves(game.board)
    probs = vec(probs)
    probs = probs .* valid_moves
    probs = probs ./ sum(policy)
    moves = Dict{Int, Float64}
    for move in 1:4096
        if probs[move] > 0
            moves[move] = probs[move]
        end
    end
    return (moves, argmax(moves[:]))
end
