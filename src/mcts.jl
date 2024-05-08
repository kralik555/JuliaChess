using Chess
using SparseArrays
include("model.jl")
include("board_class.jl")


mutable struct Node
	game::SimpleGame
	args::Dict
	parent::Union{Node, Nothing}
	action_taken::Union{Int, Nothing}
	prior::Float64
	children::Vector{Node}
	visit_count::Integer
	value_sum::Float32
	pruned_children::Vector{Node}
    q_value::Float64

	Node(game, args, parent=nothing, action_taken=nothing, prior=0.0, pruned_children=[], q_value=0.0) = 
	new(game, args, parent, action_taken, prior, [], 0, 0.0, [], 10 + prior)
end


function is_fully_expanded(node::Node)
	return length(node.children) > 0
end


function get_ucb(child::Node, node::Node)
	return child.q_value + node.args["C"] * sqrt(2*log(node.visit_count + 1)/(child.visit_count + 1)) * child.prior
end



function select(node::Node)
	best_child = Nothing
	best_ucb = -Inf
	for child in node.children
		ucb = get_ucb(child, node)
		if ucb >= best_ucb
			best_child = child
			best_ucb = ucb
		end
	end
	return best_child
end

function expand(node::Node, policy)
	for move_idx in eachindex(policy)
		prob = policy[move_idx]
		if prob > 0
			move = int_to_move(move_idx)
			child_state = deepcopy(node.game)
            if ptype(pieceon(node.game.board, from(move))) == PAWN
                if Chess.rank(to(move)) == RANK_8 || Chess.rank(to(move)) == RANK_1
                    move = Move(move.from, move.to, QUEEN)
                end
            end
            domove!(child_state, move)
            child = Node(child_state, node.args, node, move_idx, prob)
			push!(node.children, child)
		end
	end
end


function backpropagate(node::Node, value::Union{Float32, Int64, Float64})
	node.value_sum += value
	node.visit_count += 1
	node.q_value = node.value_sum / node.visit_count
	if node.parent !== nothing
		backpropagate(node.parent, value)
	end
end


mutable struct MCTS
	game::SimpleGame
	args::Dict
	model::ChessNet
end


function MCTS(game, args, model)
	return MCTS(game, args, model)
end


function search(tree::MCTS)
	root = Node(tree.game, tree.args)
    time0 = time()
    color = sidetomove(tree.game.board)
    searches = 0
    while time() - time0 < tree.args["search_time"]	&& searches < tree.args["num_searches"]
        node = root
        value = 0.0
		while is_fully_expanded(node)
			node = select(node)
		end
		if !(isterminal(node.game))
			policy, value = tree.model.model(board_to_tensor(node.game.board))
			valid_moves = get_valid_moves(node.game.board)
			policy = vec(policy)
			policy = policy .* valid_moves
			policy = policy ./ sum(policy) 
			expand(node, policy)
			value = only(value)
        else
            value = game_result(node.game)
		end
		if color == BLACK
			value = -value
		end
		backpropagate(node, value)
        searches += 1
	end
    action_probs = spzeros(Float64, 4096)
	for child in root.children
		action_probs[child.action_taken] = child.visit_count
	end
	action_probs = action_probs ./ sum(action_probs)
    print(searches, " ", time() - time0, " ")
    return action_probs, root.value_sum / root.visit_count
end

function prune_worst_branches(node::Node, max_pruned::Int64)
	children_value = Vector{Pair{Node, Float64}}()
	children_value = [Pair{Node, Float64}(child, child.value_sum/child.visit_count) for child in node.children]
	sort!(children_value, by=x->x.second)

    pruned_nodes = map(first, children_value[1:max_pruned])
	node.pruned_children = pruned_nodes

	node.children = map(first, children_value[max_pruned+1:end])
end

function search_action_reduction(tree::MCTS)
	root = Node(tree.game, tree.args)
    time0 = time()
    color = sidetomove(tree.game.board)
    searches = 0
    while time() - time0 < tree.args["search_time"]	&& searches < tree.args["num_searches"]
		if searches == length(root.children) * 2
			prune_worst_branches(root, div(length(root.children), 2))
		end
        node = root
        value = 0.0
		while is_fully_expanded(node)
			node = select(node)
		end
		if !(isterminal(node.game))
			policy, value = tree.model.model(board_to_tensor(node.game.board))
			valid_moves = get_valid_moves(node.game.board)
			policy = vec(policy)
			policy = policy .* valid_moves
			policy = policy ./ sum(policy) 
			expand(node, policy)
			value = only(value)
        else
            value = game_result(node.game)
		end
		if color == BLACK
			value = -value
		end
		backpropagate(node, value)
        searches += 1
	end
	action_probs = zeros(Float64, 4096)
	for child in root.children
		action_probs[child.action_taken] = child.visit_count
	end
	action_probs = action_probs ./ sum(action_probs)
    print(searches, " ", time() - time0, " ")
    return action_probs, root.value_sum / root.visit_count
end
