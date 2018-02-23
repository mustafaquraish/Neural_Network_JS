let training_data = [{
    inputs: [0, 0],
    outputs: [0]
  },
  {
    inputs: [0, 1],
    outputs: [1]
  },
  {
    inputs: [1, 0],
    outputs: [1]
  },
  {
    inputs: [1, 1],
    outputs: [0]
  }
];

var nn;

function setup() {

  nn = new NeuralNetwork(3,[2,3,1]);

  for (let i = 0; i < 1000000; i++) {
    let data = random(training_data);
    nn.train(data.inputs, data.outputs);
  }

  console.log(nn.guess([1,1]));
  console.log(nn.guess([0,0]));
  console.log(nn.guess([1,0]));
  console.log(nn.guess([0,1]));
}
