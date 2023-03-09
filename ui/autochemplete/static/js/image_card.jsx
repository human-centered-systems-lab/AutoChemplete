import React from "react";
import {
  Card,
  CardContent,
  Typography,
  Box,
  CardMedia
} from "@material-ui/core";

const InputImageCard = (props) => {
  const { headerImage, imageAltText } = props;

  return (
    <Card style={{ maxWidth: 300 }}>
      <CardMedia
        component="img"
        image={headerImage}
        title={imageAltText}
      />

      <CardContent>
        <Typography>Input image to label</Typography>
        <Box minHeight="0.5vh"></Box>
      </CardContent>
    </Card>
  );
};

export default InputImageCard;
