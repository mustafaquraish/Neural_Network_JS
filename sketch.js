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

  var dataFileBuffer  = loadStrings('./mnist/train-images-idx3-ubyte');
  var labelFileBuffer = loadStrings('./mnist/train-labels-idx1-ubyte');
  var pixelValues     = [];

  // It would be nice with a checker instead of a hard coded 60000 limit here
  for (var image = 0; image <= 10; image++) {
    var pixels = [];

    for (var x = 0; x <= 27; x++) {
      for (var y = 0; y <= 27; y++) {
        pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
      }
    }

    var imageData  = {};
    imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

    pixelValues.push(imageData);
  }

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
