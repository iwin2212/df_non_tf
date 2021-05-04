function get_brand(en){
    name_en = en.name
    // console.log(name_en.slice(name_en.lastIndexOf('/')+1, -4))
    // console.log(en.parentElement.parentElement.parentElement.parentNode)
    img = en.parentElement.parentElement.children[0].children[0].children[0];
    if (en.value == ''){
        $(img).hide();
    }
    else{
        console.log(img)
        $(img).show();
    }
}

function save(){
    $('#structure').submit();
}