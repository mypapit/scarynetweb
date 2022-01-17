const CLASS_LABEL = Array("non-scary", "scary");

const fileInput = document.getElementById("file-input");
const image = document.getElementById("image");
const result = document.getElementById("prediction");
const explain = document.getElementById("fuzzyexplain");

fileInput.addEventListener("change", getImage);

// Async loading

model = "";
(async () => {
  console.log("before start");

  document.getElementById("loader").style.display = "block";

  model = await tf.loadGraphModel("scaryjs_model/model.json");
  //model.summary();
  console.log("after start");
  document.getElementById("loader").style.display = "none";

  // Remove loading class from body
  document.body.classList.remove("loading");

  // When user uploads a new image, display the new image on the webpage
  fileInput.addEventListener("change", getImage);
})();

function getImage() {
  // Check if an image has been found in the input
  if (!fileInput.files[0]) throw new Error("Image not found");
  const file = fileInput.files[0];

  // Get the data url form the image
  const reader = new FileReader();

  // When reader is ready display image
  reader.onload = function(event) {
    // Ge the data url
    const dataUrl = event.target.result;

    // Create image object
    const imageElement = new Image();
    imageElement.src = dataUrl;

    // When image object is loaded
    imageElement.onload = function() {
      // Set <img /> attributes
      image.setAttribute("src", this.src);
      image.setAttribute("height", this.height);
      image.setAttribute("width", this.width);

      document.getElementById("loader").style.display = "block";
      console.log("height " + this.height);

      // Classify image
      classifyImage();
    };

    // Add the image-loaded class to the body
    document.body.classList.add("image-loaded");
  };

  // Get data URL
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", getImage);

function classifyImage() {
  tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .expandDims();

  (async () => {
    predictions = await model.predict(tensor).data();
    top2 = Array.from(predictions)
      .map(function(p, i) {
        return {
          probability: p,
          className: CLASS_LABEL[i],
          classnum: i
        };
      })
      .sort(function(a, b) {
        return b.probability - a.probability;
      })
      .slice(0, 5);

    result.innerHTML = "";
    top2.forEach(function(p) {
      console.log(p.className + ":" + p.probability.toFixed(4));
      result.innerHTML =
        result.innerHTML +
        p.className +
        ":" +
        p.probability.toFixed(4) +
        "<br />";
    });
    console.log("\n");
    if (top2[0].classnum > 0) {
      explain.innerHTML =
        "This image has been blurred as it contains disturbing or scary elements according to ScaryNet. Tap on the image to unblur it.";
      image.classList.toggle("blur", true);
    } else {
      explain.innerHTML = "This is a non-scary image according to ScaryNet.";
      image.classList.toggle("blur", false);
    }
    document.getElementById("loader").style.display = "none";
  })();
}

function toggleBlur(obj) {
  obj.classList.toggle("blur");
}

function ddx() {
  if (navigator.share) {
    navigator
      .share({
        title: "ScaryNet",
        text: "Compact Neural network for detecting scary images",
        url: window.location.href
      })
      .then(() => {
        //alert('Thanks for sharing!');
      })
      .catch(err => {
        alert("Couldn't share because " + err.message);
      });
  } else {
    //alert('Web share not supported, please use compatible device!');
  }
}

function dprobability() {
  result.classList.toggle("showprobability");
}
