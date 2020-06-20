var img, img_x, img_y, angle, rotation, lastMove, moveset, movecount, gamecount;

function setup() {
  createCanvas(300, 250);

  var imgNo = Math.floor(Math.random() * (16 - 1) + 1);
  img = loadImage('assets/animals/'+imgNo+'.png');

  img_x = 0;
  img_y = 0;
  rotation = 360/7;
  angle = rotation * Math.floor(Math.random() * (7 - 0) + 1);

  buttonAnti = createButton('Anti');
  buttonAnti.position(250, 125);
  buttonAnti.mousePressed(rotateAnti);

  buttonClock = createButton('Clock');
  buttonClock.position(25, 125);
  buttonClock.mousePressed(rotateClock);

  buttonDone = createButton('Done');
  buttonDone.position(125, 225);
  buttonDone.mousePressed(done);

  gamecount = 1;
  lastMove = createDiv('Fresh Game');
  moveset = [];
  movecount = 1;
  
  imageMode(CENTER);
  angleMode(DEGREES);
}

function draw() {
  background(255);
  noFill();
  rect(0, 0, 300, 250);
  strokeWeight(2);
  stroke(51);
  circle(150, 125, 115);

  drawImage();
}

function drawImage(){
  translate(width / 2, height / 2);
  rotate(angle);
  image(img, img_x, img_y);
}

function rotateAnti(){

  lastMove.remove();
  lastMove = createDiv('Last Move: Anticlockwise');

  save('anti_game_'+gamecount+'_move_'+movecount+'.png');
  moveset.push(-1);
  movecount ++;

  angle -= rotation;
}

function rotateClock(){
  lastMove.remove();
  lastMove = createDiv('Last Move: Clockwise');

  save('clock_game_'+gamecount+'_move_'+movecount+'.png');
  moveset.push(1);
  movecount ++;

  angle += rotation;
}

function done(){
  lastMove.remove();
  lastMove = createDiv('Finished');

  save('done_game_'+gamecount+'_move_'+movecount+'.png');
  moveset.push(0);
  movecount ++;

  // saveGameLog();
  reset();
}

function saveGameLog(){
  let writer = createWriter('game_'+gamecount+'_log.txt');
  for (var i = 0; i<moveset.length; i++){
    writer.write(moveset[i]);
    writer.write('\n');
  }
  writer.close();
}

function reset(){
  gamecount ++;
  moveset = [];
  movecount = 1;

  imgNo = Math.floor(Math.random() * (16 - 1) + 1);
  img = loadImage('assets/animals/'+imgNo+'.png');

  img_x = 0;
  img_y = 0;
  rotation = 360/7;
  angle = rotation * Math.floor(Math.random() * (7 - 0) + 1);
  
  lastMove.remove();
  lastMove = createDiv('Fresh Game');
}