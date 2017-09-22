var net;

var obstacleType = {
    "NONE": -1,
    "CACTUS_LARGE": 0,
    "CACTUS_SMALL": 1,
    "PTERODACTYL": 2
};

// obstacle[0]'s xPos yPos type cur_speed, cur_state
var num_inputs = 5;
// jump, duck, do nothing
var num_actions = 3;
// agent live in moment
var temporal_window = 0;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

// two hidden layer
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'regression', num_neurons:num_actions});

// options for the Temporal Difference learner that trains the above net
// by backpropping the temporal difference learning rule.
var tdtrainer_options = {
    learning_rate: 0.1,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 100;
opt.gamma = 0.89;
opt.learning_steps_total = 20000;
opt.learning_steps_burnin = 300;
opt.epsilon_min = 0;
opt.epsilon_test_time = 0.05;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

var brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

var runner;

function getInputValue() {
    var obstacles = runner.horizon.obstacles;
    if (obstacles.length == 0) {
        return -1;
    }
    var obstacle = obstacles[0];
    var xPos = obstacle.xPos;
    var yPos = obstacle.yPos;
    var t = obstacle.typeConfig.type;
    var state = runner.tRex.jumping << 1 | runner.tRex.ducking;
    return [xPos, yPos, obstacleType[t], runner.currentSpeed, state];
}

function periodic() {
    if (runner.crashed) {
        brain.backward(-1000);
        runner.restart();
        return;
    }
    if (!runner.playing) {
        runner.loadSounds();
        runner.playing = true;
        runner.update();
        return;
    }
    input = getInputValue()
    if (input === -1) {
        return;
    }
    brain.backward(1);
    var action = brain.forward(input)
    // 0: do nothing, 1: jump, 2: sqant
    if (action == 1) {
        if (runner.tRex.ducking) {
            runner.tRex.setDuck(false);
        }
        if (!runner.tRex.jumping && !runner.tRex.ducking) {
            runner.tRex.startJump(runner.currentSpeed)
        }
    } else if (action == 2) {
        if (runner.tRex.jumping) {
            runner.tRex.setSpeedDrop();
        } else if (!runner.tRex.jumping && !runner.tRex.ducking) {
            runner.tRex.setDuck(true);
        }
    } else if (action == 0) {
        if (runner.tRex.ducking) {
            runner.tRex.setDuck(false);
        }
    }
}

function start() {
    runner = window["Runner"].instance_
    net = new convnetjs.Net();
    setInterval(periodic, 150);
}
