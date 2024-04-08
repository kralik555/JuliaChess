using LinearAlgebra
using MKL
using Chess
using Chess.PGN
using Chess.UCI
using SparseArrays
using Serialization
include("board_class.jl")


function most_common_positions(file_path::String, save_path::String, move_depth::Int=8, max_searched::Int=100000)
    position_counter = Dict{String, Int}()
    games_played = 0 
	for game in gamesinfile(file_path)
		games_played += 1
		board = startboard()
       	moves = 0
		is_fine = true
		for move in game.:history
			if move.move == nothing || isterminal(board)
				is_fine = false
				break
			end
			domove!(board, move.move)
			moves += 1
			if moves == move_depth
				break
			end
		end
		if is_fine == true
			position_counter[fen(board)] = get!(position_counter, fen(board), 0) + 1	
			println(fen(board))
		end
		if games_played == max_searched
			break
		end
	end
    # Sort positions by most common and take the top 100
    most_common = sort(collect(position_counter), by=x->x[2], rev=true)[1:100]
    
    # Save to file
    open(save_path, "w") do file
        for (position, _) in most_common
            write(file, position * "\n")
        end
    end
	println(most_common)
end


function load_most_common(path::String)
    positions = Vector{String}()
    open(path, "r") do file
        for line in eachline(file)
            push!(positions, line)
        end
    end
    return positions
end


function get_values_policies(file_path::String, save_path::String, stockfish_path::String)
	file_number = 1
	engine = runengine(stockfish_path)
	visited_positions = Vector{String}()
	pos_dict = Dict{String, Tuple{SparseVector{Float64}, Float64}}()
	i = 1
	for game in gamesinfile(file_path)
		println(i)
		i += 1
		board = startboard()
		setboard(engine, board)
		for m in game.:history
			move = m.move
			if typeof(move) == Nothing || isterminal(game)
				break
			end
			f = fen(board)
			move_values = Vector{Float64}()
			move_indexes = Vector{Int}()
			if !(f in visited_positions)
				push!(visited_positions, f)
				# add the stockfish evaluation to the dictionary
				engine_move_values = mpvsearch(board, engine, depth=8)
				move_values = Vector{Float64}()
				move_indexes = Vector{Int}()
				for move_value in engine_move_values
					move_repr = move_value.pv[1]
					value = move_value.score.value
					if move_value.score.ismate == true
						value = 1500
						if sidetomove(board) == BLACK
							value = -1500
						end
					end
					push!(move_indexes, encode_move(tostring(move_repr)))
					push!(move_values, value)
				end

				position_value = 0.0
				if sidetomove(board) == WHITE
					position_value = maximum(move_values)
				else
					position_value = minimum(move_values)
				end
				position_value = change_value(position_value)
				changed_move_values = change_policy(move_values)
				policy = SparseVector(4096, move_indexes, changed_move_values)

				pos_dict[f] = (policy, position_value)
			end
			domove!(board, move)
		end
		if length(collect(keys(pos_dict))) >= 1000
			serialize("$(save_path)chunk_$(file_number).bin", pos_dict)
			pos_dict = Dict{String, Tuple{SparseVector{Float64}, Float64}}()
			file_number += 1
		end
	end
end



if abspath(PROGRAM_FILE) == @__FILE__
	get_values_policies("../data/files/data_2016_02.pgn", "../data/chunks/", "../stockfish/stockfish")
end
