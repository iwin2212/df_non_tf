document.addEventListener('keydown', function (event) {
    if (event.code === 'Space') {
        $.post('./snap_shot', function (data, status) {
            if (data['result'] == true) {
                $("#saving_action").show();
                var timeleft = 1;
                var downloadTimer = setInterval(function () {
                    timeleft -= 1;
                    if (timeleft <= 0) {
                        clearInterval(downloadTimer);
                        $("#saving_action").hide();
                    }
                }, 1000);
            }
            else {
                console.log('Error')
            }
        });
    }
    else if (event.key == "Escape") {
        window.location.href = "./";
    }
    else if (event.key == "Enter") {
        window.location.href = "./brandname";
    }
});