function get_brand(en) {
  try {
    v_img = en.parentElement.children[0];
    if (en.value == '') {
      $(v_img).hide();
    }
    else {
      $(v_img).show();
    }
  } catch (error) {
  }
}

function delete_img(en) {
  entity = en.parentElement.children[2];
  entity.value = "delete";
  console.log(entity)
  v_img = entity.parentElement.children[0];
  img_block = en.parentElement.parentElement.parentElement;
  if (entity.value == '') {
    $(img_block).show();
    $(v_img).hide();
  }
  else {
    $(v_img).hide();
    $(img_block).hide();
  }
}

function rename() {
  show_loading();
  rename_list = {};
  list_unknown_img.forEach(function (i) {
    new_name = document.getElementById(i).value.trim();
    if (new_name != '') {
      rename_list[i] = new_name;
    }
  });

  console.log(Object.keys(list_unknown_img).length);
  console.log(Object.keys(rename_list));
  console.log(Object.keys(rename_list).length);
  console.log(JSON.stringify(rename_list))
  $.post('./readdress?rename_list=' + JSON.stringify(rename_list), function (data, status) {
    hide_loading();
  });
  window.location.href = "./brandname";
}

function show_loading() {
  $("#loading").show();
  $("#rename").hide();
}

function hide_loading() {
  $("#loading").hide();
  $("#rename").show();
}

$('.owl-nav').ready(function (event) {
  $('.owl-prev span').hide();
});

