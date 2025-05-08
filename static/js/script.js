document.getElementById('fileInput').addEventListener('change', function () {
  var file = this.files[0];
  var fileName = file.name;
  var audioPreview = document.getElementById('audioPreview');
  var audioSource = document.getElementById('audioSource');

  var formData = new FormData();
  formData.append('file', file);

  fetch('/upload', {
    method: 'POST',
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }
      console.log('File uploaded successfully');

      // Update audio source to use server endpoint
      audioSource.src = '/get_original_audio';
      audioPreview.load();

      document.getElementById('fileName').innerText = fileName;
      document.getElementById('uploadedFile').style.display = 'flex';
      document.getElementById('uploadArea').style.display = 'none';
      document.getElementById('buttonContainer').style.display = 'flex';
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('Error uploading file. See console for details.');
    });
});

document.getElementById('predictButton').addEventListener('click', function () {
  // Show progress bar
  document.getElementById('progressBarFill').style.width = '0%';
  document.getElementById('progressContainer').style.display = 'block';

  // Use fetch API instead of XMLHttpRequest for consistency
  fetch('/predict', {
    method: 'POST',
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide progress container
      document.getElementById('progressContainer').style.display = 'none';

      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }

      // Update segments dropdown
      var $segments = document.getElementById('segments');
      $segments.innerHTML = '';
      data.segmentTimes.forEach(function (time, index) {
        var option = document.createElement('option');
        option.value = index;
        option.text = 'Segment ' + (index + 1) + ' (' + time.toFixed(2) + 's)';
        $segments.appendChild(option);
      });

      // Show segment dropdown
      document.getElementById('segment-dropdown').style.display = 'block';

      // Reset current stem info if it exists
      if (document.getElementById('currentStemInfo')) {
        document.getElementById('currentStemInfo').textContent = '';
      }
    })
    .catch((error) => {
      document.getElementById('progressContainer').style.display = 'none';
      console.error('Error:', error);
      alert('Error predicting. See console for details.');
    });
});

document.getElementById('view-segment').addEventListener('click', function () {
  var index = document.getElementById('segments').value;

  fetch('/get_segment_image?index=' + index)
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('segment-image').innerHTML = '<img src="data:image/png;base64,' + data.image + '">';
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('Error getting segment image. See console for details.');
    });
});

document.getElementById('newUploadButton').addEventListener('click', function () {
  document.getElementById('fileInput').value = '';
  document.getElementById('uploadedFile').style.display = 'none';
  document.getElementById('uploadArea').style.display = 'block';
  document.getElementById('buttonContainer').style.display = 'none';
  document.getElementById('segmentButtonsContainer').style.display = 'none';
  document.getElementById('plot').style.display = 'none';
  document.getElementById('progressContainer').style.display = 'none';
  document.getElementById('segment-dropdown').style.display = 'none';
  document.getElementById('segment-image').innerHTML = '';
  document.getElementById('stemSelect').innerHTML = '';
  document.getElementById('stemsContainer').style.display = 'none';
  document.getElementById('currentStemInfo').textContent = '';

  // Reset audio preview
  var audioPreview = document.getElementById('audioPreview');
  audioPreview.pause();
  audioPreview.currentTime = 0;
  document.getElementById('audioSource').src = '';

  document.getElementById('fileInput').click();
});

var audioPreview = document.getElementById('audioPreview');
audioPreview.addEventListener('timeupdate', function () {
  var progressBar = document.getElementById('progressBar');
  var progress = (audioPreview.currentTime / audioPreview.duration) * 100;
  progressBar.style.width = progress + '%';
});

audioPreview.addEventListener('ended', function () {
  document.getElementById('playButton').innerHTML = '▶';
  document.getElementById('progressBar').style.width = '0%';
});

document.getElementById('splitStemsButton').addEventListener('click', function () {
  document.getElementById('progressBarFill').style.width = '0%';
  document.getElementById('progressContainer').style.display = 'block';

  fetch('/split_stems', {
    method: 'POST',
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('progressContainer').style.display = 'none';

      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }

      var stemSelect = document.getElementById('stemSelect');
      stemSelect.innerHTML = '';

      // Add audio elements for each stem
      var stemAudioContainer = document.getElementById('stemAudioContainer');
      stemAudioContainer.innerHTML = ''; // Clear existing audio elements

      data.stems.forEach(function (stem) {
        // Add to select
        var option = document.createElement('option');
        option.value = stem;
        option.text = stem.charAt(0).toUpperCase() + stem.slice(1);
        stemSelect.appendChild(option);

        // Create audio player for stem
        var audioDiv = document.createElement('div');
        audioDiv.className = 'stem-audio-player';

        var label = document.createElement('div');
        label.className = 'stem-label';
        label.textContent = stem.charAt(0).toUpperCase() + stem.slice(1);

        var audio = document.createElement('audio');
        audio.controls = true;
        audio.src = '/get_stem_audio/' + stem;

        audioDiv.appendChild(label);
        audioDiv.appendChild(audio);
        stemAudioContainer.appendChild(audioDiv);
      });

      document.getElementById('stemsContainer').style.display = 'flex';
      document.getElementById('stemsContainer').style.alignItems = 'center';
      document.getElementById('stemsContainer').style.flexDirection = 'column';
    })
    .catch((error) => {
      document.getElementById('progressContainer').style.display = 'none';
      console.error('Error:', error);
      alert('Error splitting stems. See console for details.');
    });
});

// Handle stem prediction
document.getElementById('predictStemButton').addEventListener('click', function () {
  var selectedStem = document.getElementById('stemSelect').value;

  // Show progress
  document.getElementById('progressBarFill').style.width = '0%';
  document.getElementById('progressContainer').style.display = 'block';

  fetch('/predict_stem', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      stem: selectedStem,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Hide progress
      document.getElementById('progressContainer').style.display = 'none';

      if (data.error) {
        alert('Error: ' + data.error);
        return;
      }

      // Update segments dropdown
      var $segments = document.getElementById('segments');
      $segments.innerHTML = '';
      data.segmentTimes.forEach(function (time, index) {
        var option = document.createElement('option');
        option.value = index;
        option.text = 'Segment ' + (index + 1) + ' (' + time.toFixed(2) + 's)';
        $segments.appendChild(option);
      });

      // Show current stem info
      document.getElementById('currentStemInfo').textContent = 'Currently predicting: ' + selectedStem + ' stem';

      // Show segment dropdown
      document.getElementById('segment-dropdown').style.display = 'block';

      // Clear any previous images
      document.getElementById('segment-image').innerHTML = '';
    })
    .catch((error) => {
      document.getElementById('progressContainer').style.display = 'none';
      console.error('Error:', error);
      alert('Error predicting stem. See console for details.');
    });
});
