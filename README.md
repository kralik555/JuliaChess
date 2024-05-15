# Using Monte Carlo Tree Search To Play Chess

## Usage
To run any of the code, you need to install the Julia programming language and some of its packages. The packages required to run the code are Flux, Chess and MLK. Other packagees should be installed directly into the Julia distribution, but I might have forgotten to mention some here, so install any packages required that pop up when running the program.

To train the models more, run either supervised_trainin.jl or self_play.jl scripts in the src directory. 

To play against the model, run the test.jl script. Here the moves you want to play are accepted in the command line in format "{from square}{to square}", so move from e2 to e4 is simply "e2e4".
There is currently no GUI implementation for playing as it was not part of the thesis and there was not enough time to implement upgrades like this.

If you want to create a bigger database of evaluated positions, run function create_dataset in src/supervised_training.jl that runs stockfish to evaluate games from a database instead of the currently present train_on_dataset. 

To run this code, you will need stockfish downloaded and paste the path to it to the function. This also requires you to have a lichess database downloaded. Paste the path to the database into the function as well. Databases can be downloaded at https://database.lichess.org/. The dataset used in this thesis was from February 2016.
