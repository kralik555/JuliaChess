using Flux
using Chess
using Chess.UCI
using Chess.PGN
using JLD2
using Flux.Data: DataLoader
using LinearAlgebra
include("data_reader.jl")
include("model.jl")
include("board_class.jl")

# ==============================================
function train_batch(model::ChessNet, tensors, move_distros, game_values, opt)
	function loss(x, y_moves, y_value)
		y_pred_moves, y_pred_value = model.model(x)
 		move_loss = Flux.kldivergence(y_pred_moves, y_moves)
        value_loss = Flux.mse(y_pred_value, y_value)
		println("Value loss: ", 40 * value_loss, " Policy loss: ", move_loss)
        return move_loss + 40 * value_loss
	end

	
	tensors = Float32.(tensors)
	move_distros = Float32.(move_distros)
	game_values = Float32.(game_values)
	game_values = reshape(game_values, 1, :)
	#move_distros = permutedims(move_distros, (2, 1))

    println(size(move_distros))
    println(size(game_values))
    println(size(tensors))
	data_loader = DataLoader((tensors, move_distros, game_values), batchsize=256, shuffle=true)
	
	for (x_batch, y_move_batch, y_value_batch) in data_loader
		x_batch = Float32.(x_batch)
		Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)], opt)
	end
	return model
end


function train_on_dict(model::ChessNet, file_path::String, num_epochs::Int, opt)
	for epoch in 1:num_epochs
		println("Epoch ", epoch)
		files = readdir(file_path)
		num_chunks = size(files)[1]
		for i in 1:num_chunks
			println("Chunk ", i)
			tensors = []
			move_distros = []
			game_values = []
			chunk = deserialize("$(file_path)chunk_$i.bin")
			k = collect(keys(chunk))
			for key in k
				board = fromfen(key)
				tensor = create_input_tensors(board)
				move_distr = Vector(chunk[key][1])
				value = chunk[key][3] / chunk[key][2]
				push!(tensors, tensor)
				push!(move_distros, reshape(move_distr, 1, :))
				push!(game_values, value)
			end
			tensors = permutedims(cat(tensors..., dims=4), (2, 3, 1, 4))
			move_distros = vcat(move_distros...)
			println("Got to training!")
			model = train_batch(model, tensors, move_distros, game_values, opt)
		end

		model_save_path = "../models/supervised_model_$(epoch).jld2"
		JLD2.@save model_save_path model                                            
	end
end


function train_with_stockfish(model::ChessNet, stockfish_path::String)
	engine = runengine(stockfish_path)
	positions = Vector{String}()
	values = Vector{Float64}()
	policies = Vector{SparseVector{Float64}}()
	opt = Adam(0.001)

	for i in 1:10_000
		board = startboard()
		setboard(engine, board)
        println(i)
        if i == 500
            opt.eta = 0.0001
        elseif i == 2000
            opt.eta = 0.0001
        end
        played_moves = 0
		while !isterminal(board)
			push!(positions, fen(board))
			engine_move_values = mpvsearch(board, engine, depth=10)
			move_values = Vector{Float64}()
			move_indexes = Vector{Int}()
			for move_value in engine_move_values
                move = move_value.pv[1]
				value = move_value.score.value
				if move_value.score.ismate == true
					value = 1500
					if move_value.score.value == -1
						value = -1500
					end
				end
				push!(move_indexes, encode_move(tostring(move)))
				push!(move_values, value)
			end
			position_value = 0.0
			if sidetomove(board) == WHITE
				position_value = maximum(move_values)
			else
				position_value = minimum(move_values)
                move_values *= -1
			end
			position_value = change_value(position_value)
			changed_move_values = change_policy(move_values, board)
			# policy and value changed to be between 0 and 1 and policy to make a probability distribution
			policy = SparseVector(4096, move_indexes, changed_move_values)
			domove!(board, rand(moves(board)))
			push!(values, position_value)
			push!(policies, policy)
			if length(positions) == 256
				position_tensors = []
				for position in positions
					push!(position_tensors, create_input_tensors(fromfen(position)))
				end
				v_policies = Vector{Any}()
				for policy in policies
					v_policy = Vector(policy)
					push!(v_policies, reshape(v_policy, 1, :))
				end
				v_policies = vcat(v_policies...)
				position_tensors = permutedims(cat(position_tensors..., dims=4), (2, 3, 1, 4))
				model = train_batch(model, position_tensors, v_policies, values, opt)
				JLD2.@save "../models/random_stockfish_different_policy_v2.jld2" model
				positions = Vector{String}()
				values = Vector{Float64}()
				policies = Vector{SparseVector{Float64}}()
			end
            played_moves += 1
            if played_moves >= 100
                break
            end
		end
	end
	quit(engine)
end


function train_with_stockfish_on_dataset(model::ChessNet, stockfish_path::String, file_path::String)
	engine = runengine(stockfish_path)
	positions = Vector{String}()
	values = Vector{Float64}()
	policies = Vector{SparseVector{Float64}}()
	opt = Adam(0.01)
    i = 1
    for game in gamesinfile(file_path)
		board = startboard()
		setboard(engine, board)
        println(i)
        i += 1
        if i == 500
            opt.eta = 0.001
        elseif i == 2000
            opt.eta = 0.0001
        end
        played_moves = 0
		for m in game.:history
            move = m.move
			if typeof(move) == Nothing || isterminal(game)
				break
			end
			push!(positions, fen(board))
			engine_move_values = mpvsearch(board, engine, depth=8)
			move_values = Vector{Float64}()
			move_indexes = Vector{Int}()
			for move_value in engine_move_values
                move = move_value.pv[1]
				value = move_value.score.value
				if move_value.score.ismate == true
					value = 1500
					if move_value.score.value == -1
						value = -1500
					end
				end
				push!(move_indexes, encode_move(tostring(move)))
				push!(move_values, value)
			end
			position_value = 0.0
			if sidetomove(board) == WHITE
				position_value = maximum(move_values)
			else
				position_value = minimum(move_values)
                move_values *= -1
			end

			position_value = change_value(position_value)
			changed_move_values = change_policy(move_values, board)
			policy = SparseVector(4096, move_indexes, changed_move_values)
            move = m.move
            domove!(board, move)
			push!(values, position_value)
			push!(policies, policy)

			if length(positions) >= 128
				position_tensors = []
				for position in positions
					push!(position_tensors, create_input_tensors(fromfen(position)))
				end
				v_policies = Vector{Any}()
				for policy in policies
					v_policy = Vector(policy)
					push!(v_policies, reshape(v_policy, 1, :))
				end
				v_policies = vcat(v_policies...)
				position_tensors = permutedims(cat(position_tensors..., dims=4), (2, 3, 1, 4))
				model = train_batch(model, position_tensors, v_policies, values, opt)
				JLD2.@save "../models/stockfish_dataset_new_policy.jld2" model
				positions = Vector{String}()
				values = Vector{Float64}()
				policies = Vector{SparseVector{Float64}}()
			end
		end
	end
	quit(engine)
end

function create_dataset(file_path::String, stockfish_path::String)
    function add_score(action_info)
        searchinfo = parsesearchinfo(action_info)
        if searchinfo.depth != 10
            return
        end
        s = searchinfo.score
        if s.ismate == true
            if s.value < 0
                push!(values, -1)
                return
            else
                push!(values, 1)
                return
            end
        end
        value = max(s.value, -1500)
        value = min(s.value, 1500)
        push!(values, value/1500)
    end

    engine = runengine(stockfish_path)
    states = Vector{String}()
    moves = Vector{Integer}()
    values = Vector{Float32}()
    for game in gamesinfile(file_path)
        board = startboard()
        for m in game.:history
            setboard(engine, board)
            move = m.move
            if typeof(move) == Nothing || isterminal(board)
                break
            end
            if fen(board) in states
                domove!(board, move)
                continue
            end
            info = search(engine, "go depth 10", infoaction=add_score)
            push!(states, fen(board))
            push!(moves, encode_move(tostring(info.bestmove)))
            domove!(board, move)
            println(length(states), "\t", length(moves), "\t", length(values))
            if length(states) == 1_024_000 || length(states) == 512_000
                serialize("../data/files/evaluated_positions.bin", (states, moves, values))
            end
        end
    end
    quit(engine)
    serialize("../data/files/evaluated_positions_full_dataset.bin", (states, moves, values))
end

function train_on_created_dataset(model::ChessNet, file_path::String, num_epochs::Int64, opt::Flux.Optimise.Adam)
    states, moves, values = deserialize(file_path)
    l = length(states)
    batch_size = 256
    for epoch in 1:num_epochs
        for chunk_num in 0:div(l, batch_size) - 1
            chunk_states = states[chunk_num * batch_size + 1:(chunk_num + 1) * batch_size]
            chunk_moves = moves[chunk_num * batch_size + 1:(chunk_num + 1) * batch_size]
            chunk_values = values[chunk_num * batch_size + 1:(chunk_num + 1) * batch_size]
            model = train_model(model, chunk_states, Float64.(chunk_values), chunk_moves, opt)
            JLD2.@save "../models/no_pooling/model_$(epoch).jld2" model
            println((chunk_num + 1) * batch_size)
        end
    end
    return model
end

function get_value(comment)
    try
        if '#' in comment
            if '-' in comment
                return -1
            end
            return 1
        end
        c = comment[9:length(comment)-2]
        value = parse(Float64, c) / 100
        if value > 1
            return 1
        elseif value < -1
            return -1
        end
        return value
    catch e
        return 2
    end
end

function train_model(model::ChessNet, states::Vector{String}, values::Vector{Float64}, moves::Vector{Integer}, opt)
	function loss(x, y_moves, y_value)
		y_pred_moves, y_pred_value = model.model(x)
 		move_loss = Flux.crossentropy(y_pred_moves, y_moves)

        y_pred_value = dropdims(y_pred_value; dims=1)
        value_loss = Flux.mse(y_pred_value, y_value)
		println("Value loss: ", 40 * value_loss, " Policy loss: ", move_loss)
        return move_loss + 40 * value_loss
	end

    policies = Vector{Vector{Float32}}()
    tensors = []
    for state in states
        tensor = board_to_tensor(fromfen(state))
        push!(tensors, tensor)
    end
    for move in moves
        policy = zeros(4096)
        policy[move] = 1
        push!(policies, Float32.(policy))
    end
    tensors = permutedims(cat(tensors..., dims=4), (1, 2, 3, 4))
    policies = hcat(policies...)
    data_loader = DataLoader((tensors, policies, values), batchsize=256, shuffle=true)
	
	for (x_batch, y_move_batch, y_value_batch) in data_loader
		Flux.train!(loss, Flux.params(model.model), [(x_batch, y_move_batch, y_value_batch)], opt)
	end
	return model
end

function train_on_dataset(model::ChessNet, file_path::String, opt)
    states = Vector{String}()
    values = Vector{Float64}()
    correct_moves = Vector{Integer}()
    for epoch in 1:10
        game_num = 1
        for game in gamesinfile(file_path, annotations=true)
            sg = SimpleGame(game)
            game_node = game.root
            if length(game_node.children) > 0
                game_node = game_node.children[1]
            end
            if comment(game_node) == nothing
                continue
            end
            println(game_num)
            game_num += 1
            for move in sg.:history[2:length(sg.:history)]
                if length(game_node.children) == 0
                    break
                end
                c = comment(game_node)
                f = fen(game_node.board)
                m = move.move
                value = get_value(c)
                if value == 2
                    break
                end
                game_node = game_node.children[1]
                push!(states, f)
                push!(correct_moves, encode_move(tostring(m)))
                push!(values, value)
                if length(values) == 256
                    model = train_model(model, states, values, correct_moves, opt)
                    states = Vector{String}()
                    values = Vector{Float64}()
                    correct_moves = Vector{Integer}()
                    save_path = "../models/supervised_model_$(epoch).jld2"
                    JLD2.@save save_path model
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
	model = ChessNet()
    #JLD2.@load "../models/model_7.jld2" model
    opt = Adam(0.0001)
    train_on_created_dataset(model, "../data/files/evaluated_positions.bin", 10, opt)
end
