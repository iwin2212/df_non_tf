function get_brand(en) {
    // console.log(name_en.slice(name_en.lastIndexOf('/')+1, -4))
    // console.log(en.parentElement.parentElement.parentElement.parentNode)
    try {
        v_img = en.parentElement.parentElement.children[0].children[0].children[0];
        x_img = en.parentElement.parentElement.children[0].children[1].children[0];
        if (en.value == '') {
            $(v_img).hide();
            $(x_img).hide();
        }
        else {
            $(x_img).hide();
            $(v_img).show();
        }
    } catch (error) {
    }
}


function rename() {
    rename_list = {}
    list_unknown_img.forEach(function (i) {
        new_name = document.getElementById(i).value.trim();
        if (new_name != ''){
            if (new_name == "Ảnh này sẽ được xoá sau khi ấn lưu lại"){
                rename_list[i] = "delete";
            }
            else{
                rename_list[i] = new_name;
            }
        }
    });
    $.post('./readdress?rename_list=' + JSON.stringify(rename_list), function (data, status) {
        console.log(data);
    });
    window.location.href = "./brandname";
}

function delete_img(id) {
    document.getElementById(id).value = "Ảnh này sẽ được xoá sau khi ấn lưu lại";
    v_img = document.getElementById(id).parentElement.parentElement.children[0].children[0].children[0];
    x_img = document.getElementById(id).parentElement.parentElement.children[0].children[1].children[0];
    if (document.getElementById(id).value == '') {
        $(x_img).hide();
        $(v_img).hide();
    }
    else {
        $(v_img).hide();
        $(x_img).show();
    }
}