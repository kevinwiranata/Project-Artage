<?php
$target_dir = "uploads/";
$target_name = basename($_FILES["fileToUpload"]["name"]);
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

if (move_uploaded_file($_FILES['fileToUpload']['tmp_name'], $target_file)) {
  echo "File is valid, and was successfully uploaded.\n";
  $command = escapeshellcmd("imageResize.py $target_name");
  $output = shell_exec($command);
  
  $newfilename = "resized/" . $target_name;

  $new_command = escapeshellcmd("runnn.py $newfilename");
  $new_output = shell_exec($new_command);
  echo "<br />Expected date range is: ";
  echo $new_output;
} else {
   echo "Upload failed";
}
?>