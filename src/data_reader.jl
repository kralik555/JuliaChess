using Chess
using Chess.PGN
using JLD2
using Glob
include("board_class.jl")


function save_positions(file_path::String, num_games::Integer)
    games_data = []
    game_count = 0
    for game in gamesinfile(file_path)
        game_count += 1
	println(game_count)
	if game_count == num_games
            break
        end
        
        board = startboard()

	res = game.headers.:result
	result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
        
    for m in game.:history
		if typeof(m.move) == Nothing || isterminal(game)
			break
	    end
	    tensor = create_input_tensors(board)
	    move = tostring(m.move)
        move_uint16 = encode_move(move)
        push!(games_data, (tensor, move_uint16, result))
        domove!(board, m.move)
        end
    end
    
    existing_files = glob("data/arrays/games_data_*.jld2")
    file_num = length(existing_files) + 1
    println("Existing files: ", file_num - 1)
    save_path = "data/arrays/games_data_$(file_num).jld2"

    @save save_path data=games_data
    
    return games_data
end

function get_elo(tags, key)
    for tag in tags
        if tag.name == key
		if tag.value == "?"
			return 0
		end
		return parse(Int, tag.value)
        end
    end
    return 0
end

function save_high_rating_positions(file_path::String, num_games::Integer)
	games_data = []
	game_count = 0
    	for game in gamesinfile(file_path)
        	white_rating = get_elo(game.headers.othertags, "WhiteElo")
			black_rating = get_elo(game.headers.othertags, "BlackElo")	
		# Skip the game if either player's rating is below 2000
        	if white_rating < 2300 || black_rating < 2300
            		continue
        	end	
        	game_count += 1
		println(game_count)
		if game_count == num_games
            	break
        	end
        
        	board = startboard()

		res = game.headers.:result
		result = res == "1-0" ? 1 : res == "0-1" ? -1 : 0
        
        	for m in game.:history
			if typeof(m.move) == Nothing || isterminal(game)
					break
	    		end
	    		tensor = create_input_tensors(board)
	    		move = tostring(m.move)
            	move_uint16 = encode_move(move)
            	push!(games_data, (tensor, move_uint16, result))
				domove!(board, m.move)
        	end
		if game_count == 10000
			break	
		end	
	end
    
    existing_files = glob("data/arrays/games_data_*.jld2")
    file_num = length(existing_files) + 1
    println("Existing files: ", file_num - 1)
    save_path = "data/arrays/games_data_$(file_num).jld2"

    @save save_path data=games_data
    
    return games_data

end


function load_data(file_path::String)
	@load file_path data
	return data
end


function most_common_positions(file_path::String, save_path::String, move_depth::Int=8)
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
		if games_played == 100000
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


if abspath(PROGRAM_FILE) == @__FILE__
	save_high_rating_positions("data/files/data_2016_02.pgn", 10000)	
	#most_common_positions("data/files/data_2016_02.pgn", "data/common_games.txt")
end
