
class NeuralNetwork {

  constructor (num_layers, nodes) {
    // Check if inputs are of right dimensions
    if (num_layers != nodes.length) {
      return undefined;
    }
    // Set number of layers and nodes in each layer
    this.num_layers = num_layers;
    this.num_nodes = nodes;

    this.weights = [];
    this.biases = [];

    this.learning_rate = 0.01; // Default value

    // Default activation, Sigmoid
    this.activation = sig;

    // Derivative of sigmoid
    this.der_activation = der_sig;

    // Generate random weight and bias matrices
    for (let i = 0; i < num_layers-1; i++) {
        let rm = new Matrix(this.num_nodes[i+1],this.num_nodes[i]);
        rm.randomize();
        this.weights.push(rm);

        rm = new Matrix(this.num_nodes[i+1],1);
        rm.randomize();
        this.biases.push(rm);
    }
  }

  guess(input_data) {
    let inputs = Matrix.fromArray(input_data);

    for (let i = 0; i < this.num_layers-1; i++) {
      inputs = Matrix.multiply(this.weights[i], inputs);
      inputs.add(this.biases[i]);
      inputs.map(this.activation);
    }

    return inputs.toArray();
  }

  train(input_data, target_data) {
    // Convert targets to a matrix
    let targets = Matrix.fromArray(target_data);

    // Initiliaze array to store layer outputs
    let layer_outputs = [];
    layer_outputs.push(Matrix.fromArray(input_data));

    // Feed forward
    for (let i = 0; i < this.num_layers-1; i++) {
      layer_outputs[i+1] = Matrix.multiply(this.weights[i], layer_outputs[i]);
      layer_outputs[i+1].add(this.biases[i]);
      layer_outputs[i+1].map(this.activation);
    }

    // Output error
    let errors = new Array(this.num_layers);

    // Store it in the last index of array
    errors[this.num_layers-1] =
      (Matrix.subtract(targets, layer_outputs[this.num_layers-1]));

    // Back propagation
    for (let i = this.num_layers - 2; i >= 0; i--) {
      // Calculate gradient for current layer
      let gradients = Matrix.map(layer_outputs[i+1], this.der_activation);
      gradients.multiply(errors[i+1]);
      gradients.multiply(this.learning_rate);

      // Calculate delta
      let pt = Matrix.transpose(layer_outputs[i]);
      let delta = Matrix.multiply(gradients, pt);

      // Adjust weights and bias
      this.weights[i].add(delta);
      this.biases[i].add(gradients);

      // Update the errors for previous layer
      let t = Matrix.transpose(this.weights[i]);
      errors[i] = Matrix.multiply(t, errors[i+1]);
    }

  }
}

// Sigmoid
function sig(x) {
    return 1/(1 + Math.exp(-x));
}

// Not actual derivative, since sigmoid already applied
function der_sig(x) {
  return (x* (1 - x));
}
