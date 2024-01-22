y1 <- read.csv("KS_res_y1_L10_df6_100.csv", header = FALSE)$V1
y2 <- read.csv("KS_res_y2_L10_df6_100.csv", header = FALSE)$V1
y3 <- read.csv("KS_res_y3_L10_df6_100.csv", header = FALSE)$V1
y4 <- read.csv("KS_res_y4_L10_df6_100.csv", header = FALSE)$V1
y5 <- read.csv("KS_100_boot_asym.csv", header = FALSE)$V1

y1_rot <- read.csv("KS_rotated_y1_L10_df6_100.csv", header = FALSE)$V1
y2_rot <- read.csv("KS_rotated_y2_L10_df6_100.csv", header = FALSE)$V1
y3_rot <- read.csv("KS_rotated_y3_L10_df6_100.csv", header = FALSE)$V1
y4_rot <- read.csv("KS_rotated_y4_L10_df6_100.csv", header = FALSE)$V1

cols <- c(
  # rgb(51, 187, 238, maxColorValue = 255),#cyan
  rgb(102, 153, 204, maxColorValue = 255),#light blue
  rgb(221, 170, 51, maxColorValue = 255),#yellow
  rgb(187, 85, 102, maxColorValue = 255),#red
  "black",
  rgb(0, 68, 136, maxColorValue = 255)#blue
)

ltys <- 1:5
lwds <- c(3,3,4,3,3)
legend_lables <- c(expression(paste("C"^"M1", ", Normal errors")),
                   expression(paste("C"^"M2", ", Normal errors")),
                   expression(paste("C"^"M1", ", ", T[6], " errors")),
                   expression(paste("C"^"M1", ", ", T[6], " errors")),
                   "asymptotic distribution")

make_plot <- function(y_list, col = cols, lty = ltys, lwd = lwds,
                      ordering = 1:length(y_list), ...){
  y_list <- y_list[ordering]
  legend_lables <- legend_lables[ordering]
  cols <- cols[ordering]
  ltys <- ltys[ordering]
  lwds <- lwds[ordering]
  x <- lapply(y_list, sort)
  y <- lapply(y_list, function(y) (1:length(y))/length(y))
  plot(x[[1]], y[[1]], type = "l", col = col[1], lty = lty[1], lwd = lwds[1], ...)
  lines(x[[2]], y[[2]], type = "l", col = col[2], lty = lty[2], lwd = lwds[2])
  lines(x[[3]], y[[3]], type = "l", col = col[3], lty = lty[3], lwd = lwds[3])
  lines(x[[4]], y[[4]], type = "l", col = col[4], lty = lty[4], lwd = lwds[4])
  if(length(y_list) == 5) lines(x[[5]], y[[5]], type = "l", col = col[5],
                                lty = lty[5], lwd = lwds[5])
  legend("bottomright", legend = legend_lables, col = cols, lty = ltys,
         lwd = lwds, seg.len = 4)
}
setEPS()
postscript("PRL_sphered1.eps",width = 8, height = 6)
make_plot(list(y1, y2, y3, y4, y5), ordering = c(1,3,2,4,5),
          xlab = "Kolmogorov-Smirnov Statistic of Sphered Residuals",
          ylab = "CDF", xlim = c(0.8, 10),
          main = "CDFs for Kolmogorov-Smirnov Statistics of Sphered Residuals")
dev.off()

setEPS()
postscript("PRL_rotated1.eps",width = 8, height = 6)
make_plot(list(y1_rot, y2_rot, y3_rot, y4_rot, y5), ordering = c(1,3,2,4,5),
          xlab = "Kolmogorov-Smirnov Statistic of Rotated Residuals",
          ylab = "CDF",
          main = "CDFs for Kolmogorov-Smirnov Statistics of Rotated Residuals")
dev.off()




