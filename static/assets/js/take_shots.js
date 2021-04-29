document.addEventListener('keydown', function (event) {
    if (event.keyCode == 32) {
        $.post('./snap_shot', function (data, status) {
            console.log(data)
        });
    }
});