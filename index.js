var EventEmitter = require('events').EventEmitter
  , inherits = require('inherits')
  , POSITIONX = 0
  , POSITIONY = 1
  , SPEEDX = 2
  , SPEEDY = 3
  , ACCELERATIONX = 4
  , ACCELERATIONY = 5

require('gpu.js');

let serialMode = true;
let toggleModeBtn = document.querySelector('.tg-btn');
toggleModeBtn.addEventListener('click', function (e) {
  if (serialMode) {
    e.target.innerHTML = "To Sequential";
  } else {
    e.target.innerHTML = "To Parallel";
  }
  serialMode = !serialMode;
});

let kernelFunction = function (boids, size, sepDist, sepForce, cohDist, cohForce, aliDist
  , aliForce, speedLimit, accelerationLimit, accelerationLimitRoot, speedLimitRoot) {

  let posX = boids[this.thread.y][0], posY = boids[this.thread.y][1],
    speedX = boids[this.thread.y][2], speedY = boids[this.thread.y][3],
    accX = boids[this.thread.y][4], accY = boids[this.thread.y][5];

  let curPosX = boids[this.thread.y][0];
  let curPosY = boids[this.thread.y][1];
  let sforceX = 0, sforceY = 0, cforceX = 0, cforceY = 0, aforceX = 0, aforceY = 0, spareX, spareY;

  let distSquared;

  let count = size;

  for (let i = 0; i < size; i++) {

    if (i === this.thread.x) continue;
    spareX = curPosX - boids[i][0];
    spareY = curPosY - boids[i][1];
    distSquared = spareX * spareX + spareY * spareY

    if (distSquared < sepDist) {
      sforceX += spareX;
      sforceY += spareY;
    } else {
      if (distSquared < cohDist) {
        cforceX += spareX;
        cforceY += spareY;
      }
      if (distSquared < aliDist) {
        aforceX += boids[i][2];
        aforceY += boids[i][3];
      }
    }
  }


  let length, a, b, lo, hi;
  // Separation
  a = Math.abs(sforceX);
  b = Math.abs(sforceY);
  lo = Math.min(a, b);
  hi = Math.max(a, b);
  length = hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16;
  if (length !== 0) {
    accX += (sepForce * sforceX / length);
    accY += (sepForce * sforceY / length);
  }
  // Cohesion
  a = Math.abs(cforceX);
  b = Math.abs(cforceY);
  lo = Math.min(a, b);
  hi = Math.max(a, b);
  length = hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16;
  if (length !== 0) {
    accX -= (cohForce * cforceX / length);
    accY -= (cohForce * cforceY / length);
  }
  // Alignment
  a = Math.abs(aforceX);
  b = Math.abs(aforceY);
  lo = Math.min(a, b);
  hi = Math.max(a, b);
  length = hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16;
  if (length !== 0) {
    accX -= (aliForce * aforceX / length);
    accY -= (aliForce * aforceY / length);
  }

  let ratio;

  if (accelerationLimit > 0) {
    distSquared = accX * accX + accY * accY;
    if (distSquared > accelerationLimit) {
      a = Math.abs(accX);
      b = Math.abs(accY);
      lo = Math.min(a, b);
      hi = Math.max(a, b);
      length = hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16;

      ratio = accelerationLimitRoot / length;
      accX *= ratio;
      accY *= ratio;
    }
  }

  speedX += accX;
  speedY += accY;

  if (speedLimit > 0) {
    distSquared = speedX * speedX + speedY * speedY;
    if (distSquared > speedLimit) {
      a = Math.abs(speedX);
      b = Math.abs(speedY);
      lo = Math.min(a, b);
      hi = Math.max(a, b);
      length = hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16;
      ratio = speedLimitRoot / length;
      speedX *= ratio;
      speedY *= ratio;
    }
  }

  posX += speedX;
  posY += speedY;


  let pos, speed, acc;
  if (this.thread.x % 2 === 0) {
    pos = posX;
    speed = speedX;
    acc = accX;
  }
  else {
    pos = posY;
    speed = speedY;
    acc = accY;
  }

  if (this.thread.x === 0 || this.thread.x === 1) return pos;
  else if (this.thread.x === 2 || this.thread.x === 3) return speed;
  else return acc;

}


// let kernel = gpu.createKernel(kernelFunction).setOutput([6, BOID_AMOUNT]);

module.exports = Boids

function Boids(opts, callback) {
  if (!(this instanceof Boids)) return new Boids(opts, callback)
  EventEmitter.call(this)

  opts = opts || {}
  callback = callback || function () { }

  this.speedLimitRoot = opts.speedLimit || 0
  this.accelerationLimitRoot = opts.accelerationLimit || 1
  this.speedLimit = Math.pow(this.speedLimitRoot, 2)
  this.accelerationLimit = Math.pow(this.accelerationLimitRoot, 2)
  this.separationDistance = Math.pow(opts.separationDistance || 60, 2)
  this.alignmentDistance = Math.pow(opts.alignmentDistance || 180, 2)
  this.cohesionDistance = Math.pow(opts.cohesionDistance || 180, 2)
  this.separationForce = opts.separationForce || 0.15
  this.cohesionForce = opts.cohesionForce || 0.1
  this.alignmentForce = opts.alignmentForce || opts.alignment || 0.25
  this.isReady = true;

  // Kernel
  let gpu = new GPU();
  this.kernel = gpu.createKernel(kernelFunction).setOutput([6, opts.boids]);

  var boids = this.boids = []
  for (var i = 0, l = opts.boids === undefined ? 50 : opts.boids; i < l; i += 1) {
    boids[i] = [
      Math.random() * 25, Math.random() * 25 // position
      , 0, 0                               // speed
      , 0, 0                               // acceleration
    ];

  }

  this.on('tick', function () {
    callback(boids)
  })
}
inherits(Boids, EventEmitter)

Boids.prototype.tick = function () {
  var boids = this.boids
    , sepDist = this.separationDistance
    , sepForce = this.separationForce
    , cohDist = this.cohesionDistance
    , cohForce = this.cohesionForce
    , aliDist = this.alignmentDistance
    , aliForce = this.alignmentForce
    , speedLimit = this.speedLimit
    , accelerationLimit = this.accelerationLimit
    , accelerationLimitRoot = this.accelerationLimitRoot
    , speedLimitRoot = this.speedLimitRoot
    , size = boids.length
    , current = size
    , sforceX, sforceY
    , cforceX, cforceY
    , aforceX, aforceY
    , spareX, spareY
    , distSquared
    , currPos
    , length
    , target
    , ratio

  if (this.isReady) {
    this.isReady = false;
    if (!serialMode) {
      // Parallel by GPU
      this.boids = this.kernel(this.boids, size, sepDist, sepForce, cohDist, cohForce, aliDist
        , aliForce, speedLimit, accelerationLimit, accelerationLimitRoot, speedLimitRoot)
    }
    else {
      // Sequential
      while (current--) {
        sforceX = 0; sforceY = 0
        cforceX = 0; cforceY = 0
        aforceX = 0; aforceY = 0
        currPos = boids[current]

        target = size

        // Need Pos (2d), Speed (2d), Acc (2d)
        while (target--) {
          if (target === current) continue
          spareX = currPos[0] - boids[target][0]
          spareY = currPos[1] - boids[target][1]
          distSquared = spareX * spareX + spareY * spareY

          if (distSquared < sepDist) {
            sforceX += spareX
            sforceY += spareY
          } else {
            if (distSquared < cohDist) {
              cforceX += spareX
              cforceY += spareY
            }
            if (distSquared < aliDist) {
              aforceX += boids[target][SPEEDX]
              aforceY += boids[target][SPEEDY]
            }
          }
        }

        // Separation
        length = hypot(sforceX, sforceY)
        boids[current][ACCELERATIONX] += (sepForce * sforceX / length) || 0
        boids[current][ACCELERATIONY] += (sepForce * sforceY / length) || 0
        // Cohesion
        length = hypot(cforceX, cforceY)
        boids[current][ACCELERATIONX] -= (cohForce * cforceX / length) || 0
        boids[current][ACCELERATIONY] -= (cohForce * cforceY / length) || 0
        // Alignment
        length = hypot(aforceX, aforceY)
        boids[current][ACCELERATIONX] -= (aliForce * aforceX / length) || 0
        boids[current][ACCELERATIONY] -= (aliForce * aforceY / length) || 0
      }
      current = size

      // Apply speed/acceleration for
      // this tick
      while (current--) {
        if (accelerationLimit) {
          distSquared = boids[current][ACCELERATIONX] * boids[current][ACCELERATIONX] + boids[current][ACCELERATIONY] * boids[current][ACCELERATIONY]
          if (distSquared > accelerationLimit) {
            ratio = accelerationLimitRoot / hypot(boids[current][ACCELERATIONX], boids[current][ACCELERATIONY])
            boids[current][ACCELERATIONX] *= ratio
            boids[current][ACCELERATIONY] *= ratio
          }
        }

        boids[current][SPEEDX] += boids[current][ACCELERATIONX]
        boids[current][SPEEDY] += boids[current][ACCELERATIONY]

        if (speedLimit) {
          distSquared = boids[current][SPEEDX] * boids[current][SPEEDX] + boids[current][SPEEDY] * boids[current][SPEEDY]
          if (distSquared > speedLimit) {
            ratio = speedLimitRoot / hypot(boids[current][SPEEDX], boids[current][SPEEDY])
            boids[current][SPEEDX] *= ratio
            boids[current][SPEEDY] *= ratio
          }
        }

        boids[current][POSITIONX] += boids[current][SPEEDX]
        boids[current][POSITIONY] += boids[current][SPEEDY]
      }


    }
    this.isReady = true;
  }
  // this.emit('tick', this.boids)
}

// double-dog-leg hypothenuse approximation
// http://forums.parallax.com/discussion/147522/dog-leg-hypotenuse-approximation
function hypot(a, b) {
  a = Math.abs(a)
  b = Math.abs(b)
  var lo = Math.min(a, b)
  var hi = Math.max(a, b)
  return hi + 3 * lo / 32 + Math.max(0, 2 * lo - hi) / 8 + Math.max(0, 4 * lo - hi) / 16
}

