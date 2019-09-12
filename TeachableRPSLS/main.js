// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import CountDownTimer from './countdown';
// Number of classes to classify
const NUM_CLASSES = 5;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

const MOVES = [
  'Rock',
  'Paper',
  'Scissors',
  'Lizard',
  'Spock',
];

const RULESMATRIX= [
  ["It's a draw! Try again.", 'You win!: Paper covers Rock', 'You lose: as it always has rock crushes scissors', 'You lose: Rock crushes Lizard', 'You win!: Spock vaporizes Rock'],
  ['You lose: Paper covers Rock', "It's a draw! Try again.", 'You win!Scissors cuts Paper', 'You win!: Lizard eats Paper', 'You lose: Paper disproves Spock'],
  ['You win!: as it always has rock crushes scissors', 'You lose: Scissors cuts Paper', "It's a draw! Try again.", 'You lose: Scissors decapitates Lizard', 'You win!: Spock smashes scissors'],
  ['You win!: Rock crushes Lizard', 'You lose: Lizard eats Paper', 'You win!: Scissors decapitates Lizard', "It's a draw! Try again.", 'You lose: Lizard poisons Spock' ],
  ['You lose: Spock vaporizes Rock' , 'You win!: Paper disproves Spock', 'You lose: Spock smashes scissors', 'You win!: Lizard poisons Spock', "It's a draw! Try again."],
];

const trainButtonIds = [
  'train-rock-button',
  'train-paper-button',
  'train-scissors-button',
  'train-lizard-button',
  'train-spock-button',
];

const trainSpanIds = [
  'train-rock-span',
  'train-paper-span',
  'train-scissors-span',
  'train-lizard-span',
  'train-spock-span',
];

const gestureYouIds = [
  'rock-you',
  'paper-you',
  'scissors-you',
  'lizard-you',
  'spock-you',
];

const gestureCpuIds = [
  'rock-cpu',
  'paper-cpu',
  'scissors-cpu',
  'lizard-cpu',
  'spock-cpu',
];


class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;
    this.currentMove = -1;
    this.firstExampleTrained = false;
    this.gaming = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Get video element that will contain the webcam image
    this.video = document.getElementById('cam-video');

    // Get training buttons and info texts
    for (let i = 0; i < NUM_CLASSES; i++) {


      // Get training button
      let button = document.getElementById(trainButtonIds[i])

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => {
        this.training = i; //indicate the class that we want train
        button.innerText = `Training ${MOVES[i]}...`;
      });
      button.addEventListener('mouseup', () => {
        this.training = -1; //no button pressed
        button.innerText = `Train ${MOVES[i]}`;
      });

      // Get info text
      const infoText = document.getElementById(trainSpanIds[i])
      infoText.innerText = " No examples added";
      this.infoTexts.push(infoText);
    }

    // Create button for starting a game
    this.startButton = document.getElementById('start-game-button');
    this.startButton.onclick = () => {
      this.startGame();
    };

    this.gameStatus = document.getElementById('game-status');

    this.gestureYouImages = gestureYouIds.map((val) => {
      return document.getElementById(val);
    });

    this.gestureCpuImages = gestureCpuIds.map((val) => {
      return document.getElementById(val);
    });




    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  /**
   * Start a game of rock-paper-scissors-lizard-spock
   */
  startGame() {
    if (this.startButton.disabled) {
      return;
    }
    this.gaming = true;
    this.startButton.disabled = true;
    this.gameStatus.style.fontWeight = 'normal';
    this.countDownTimer = new CountDownTimer(5000, 20);
    this.countDownTimer.addTickFn((msLeft) => {
      this.gameStatus.innerText = (msLeft/1000).toFixed(0) +
      ' seconds left. Prepare your move!';
      let computerMove = Math.floor(Math.random()*NUM_CLASSES);
      for (let i = 0; i < NUM_CLASSES; i++) {
        this.gestureCpuImages[i].hidden = (i !== computerMove);
      }
      if (msLeft == 0) {
        this.resolveGame();
      }
    });
    this.countDownTimer.start();
  }

  /**
   * Resolve the game
   */
  resolveGame() {
    this.gaming = false;
    let computerMove = Math.floor(Math.random()*NUM_CLASSES);

    this.gameStatus.innerText = RULESMATRIX[computerMove][this.currentMove];
    this.gameStatus.style.fontWeight = 'bold';
    for (let i = 0; i < NUM_CLASSES; i++) {
      this.gestureCpuImages[i].hidden = (i !== computerMove);
      this.gestureYouImages[i].hidden = (i !== this.currentMove);
    }
    this.startButton.disabled = false;
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);
        this.currentMove = res.classIndex;
        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
          }

          // Update img if in game if the button "start a game" has been pressed
          if (this.gaming) {

            if (res.classIndex == i) {
              this.gestureYouImages[i].hidden = false;
            } else {
              this.gestureYouImages[i].hidden = true;
            }
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
        if (!this.firstExampleTrained) {
          this.firstExampleTrained = true;
          this.startButton.disabled = false;
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener('load', () => new Main());
