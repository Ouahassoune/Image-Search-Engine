document.addEventListener("change", function (event) {
    if (event.target.classList.contains("uploadProfileInput")) {
      var triggerInput = event.target;
      var currentImg = triggerInput.closest(".pic-holder").querySelector(".pic")
        .src;
      var holder = triggerInput.closest(".pic-holder");
      var wrapper = triggerInput.closest(".profile-pic-wrapper");
  
      var alerts = wrapper.querySelectorAll('[role="alert"]');
      alerts.forEach(function (alert) {
        alert.remove();
      });
  
      triggerInput.blur();
      var files = triggerInput.files || [];
      if (!files.length || !window.FileReader) {
        return;
      }
  
      if (/^image/.test(files[0].type)) {
        var reader = new FileReader();
        reader.readAsDataURL(files[0]);
  
        reader.onloadend = function () {
          holder.classList.add("uploadInProgress");
          holder.querySelector(".pic").src = this.result;
  
          var loader = document.createElement("div");
          loader.classList.add("upload-loader");
          loader.innerHTML =
            '<div class="spinner-border text-primary" role="status"><span class="sr-only">Loading...</span></div>';
          holder.appendChild(loader);
  
          setTimeout(function () {
            holder.classList.remove("uploadInProgress");
            loader.remove();
  
            var random = Math.random();
            if (random < 0.9) {
              wrapper.innerHTML +=
                '<div class="snackbar show" role="alert"><i class="fa fa-check-circle text-success"></i> Profile image updated successfully</div>';
              triggerInput.value = "";
              setTimeout(function () {
                wrapper.querySelector('[role="alert"]').remove();
              }, 3000);
            } else {
              holder.querySelector(".pic").src = currentImg;
              wrapper.innerHTML +=
                '<div class="snackbar show" role="alert"><i class="fa fa-times-circle text-danger"></i> There is an error while uploading! Please try again later.</div>';
              triggerInput.value = "";
              setTimeout(function () {
                wrapper.querySelector('[role="alert"]').remove();
              }, 3000);
            }
          }, 1500);
        };
      } else {
        wrapper.innerHTML +=
          '<div class="alert alert-danger d-inline-block p-2 small" role="alert">Please choose a valid image.</div>';
        setTimeout(function () {
          var invalidAlert = wrapper.querySelector('[role="alert"]');
          if (invalidAlert) {
            invalidAlert.remove();
          }
        }, 3000);
      }
    }
  });