cols <- c(
  rgb(102, 153, 204, maxColorValue = 255),#light blue
  rgb(221, 170, 51, maxColorValue = 255),#yellow
  rgb(187, 85, 102, maxColorValue = 255),#red
  "black",
  rgb(0, 68, 136, maxColorValue = 255)#blue
)

ltys <- 1:5
lwds <- rep(3,5)

legend_lables <- c(expression(N(A[f[b] *"," *s]^{M1}, Sigma[f[b]*"," *s])),expression(N(A[f[b] *"," *s]^{M2}, Sigma[f[b]*"," *s])),
                   expression(L(A[f[b] *"," *s]^{M1}, Sigma[f[b]*"," *s])),expression(L(A[f[b] *"," *s]^{M2}, Sigma[f[b]*"," *s])),
                   "Limit")

make_plot <- function(y_list, col = cols, lty = ltys, lwd = lwds, cex = 1,
                      ordering = 1:length(y_list), xlab = "c",
                      ylab = expression(P(KS<=c)), ...){
  y_list <- y_list[ordering]
  legend_lables <- legend_lables[ordering]
  cols <- cols[ordering]
  ltys <- ltys[ordering]
  lwds <- lwds[ordering]
  x <- lapply(y_list, sort)
  y <- lapply(y_list, function(y) (1:length(y))/length(y))
  plot(x[[1]], y[[1]], type = "l", col = cols[1], lty = ltys[1], lwd = lwds[1],
       cex.axis = 2, ylab = "", xlab = "",...)
  title(xlab = xlab, ylab = ylab, cex.lab = 2, line = 3)
  lines(x[[2]], y[[2]], type = "l", col = cols[2], lty = ltys[2], lwd = lwds[2])
  lines(x[[3]], y[[3]], type = "l", col = cols[3], lty = ltys[3], lwd = lwds[3])
  lines(x[[4]], y[[4]], type = "l", col = cols[4], lty = ltys[4], lwd = lwds[4])
  if(length(y_list) == 5) lines(x[[5]], y[[5]], type = "l", col = cols[5],
                                lty = ltys[5], lwd = lwds[5])
  legend("bottomright", legend = legend_lables, col = cols, lty = ltys,
         lwd = lwds, seg.len = 4, cex = 1.5)
}

########################## Loading data ################################ 
## KS ##
y1_ks <- read.csv("KS_res_y1_nosum_960_new_L.csv", header = FALSE)$V1
y2_ks <- read.csv("KS_res_y2_nosum_960_new_L.csv", header = FALSE)$V1
y3_ks <- read.csv("KS_res_y3_nosum_960_new_L.csv", header = FALSE)$V1
y4_ks <- read.csv("KS_res_y4_nosum_960_new_L.csv", header = FALSE)$V1

y1_rot_ks <- read.csv("KS_rotated_y1_nosum_960_new_L.csv", header = FALSE)$V1
y2_rot_ks <- read.csv("KS_rotated_y2_nosum_960_new_L.csv", header = FALSE)$V1
y3_rot_ks <- read.csv("KS_rotated_y3_nosum_960_new_L.csv", header = FALSE)$V1
y4_rot_ks <- read.csv("KS_rotated_y4_nosum_960_new_L.csv", header = FALSE)$V1
y5_asy_ks <- read.csv("KS_boot_asym1_new_L.csv", header = FALSE)$V1

setEPS()
postscript("PRD_sphered_ks_new_LL4.eps", width = 9, height = 6)
par(mar = c(4, 5.5, 2, 2))
make_plot(list(y1_ks, y2_ks, y3_ks, y4_ks), ordering = c(1,3,2,4), cex = 2 ,xlim=c(0.3,2.2),
          ylab = expression(P(K[N]<=k[N])),xlab=expression(k[N]))
dev.off()

setEPS()
postscript("PRD_rotated_ks_new_LL4.eps",width = 9, height = 6)
par(mar = c(4.5, 5.5 ,2 ,2))
make_plot(list(y1_rot_ks, y2_rot_ks, y3_rot_ks, y4_rot_ks,y5_asy_ks),xlim=c(0.23,0.91),
          ordering = c(1,3,2,4,5), cex = 2, 
          ylab = expression(P(tilde(K)[N]<=tilde(k)[N])),xlab=expression(tilde(k)[N]))
dev.off()


## CVM ##
y1_cvm <- read.csv("CVM_res_y1_nosum_960_new_L.csv", header = FALSE)$V1
y2_cvm <- read.csv("CVM_res_y2_nosum_960_new_L.csv", header = FALSE)$V1
y3_cvm <- read.csv("CVM_res_y3_nosum_960_new_L.csv", header = FALSE)$V1
y4_cvm <- read.csv("CVM_res_y4_nosum_960_new_L.csv", header = FALSE)$V1

y1_rot_cvm <- read.csv("CVM_rotated_y1_nosum_960_new_L.csv", header = FALSE)$V1
y2_rot_cvm <- read.csv("CVM_rotated_y2_nosum_960_new_L.csv", header = FALSE)$V1
y3_rot_cvm <- read.csv("CVM_rotated_y3_nosum_960_new_L.csv", header = FALSE)$V1
y4_rot_cvm <- read.csv("CVM_rotated_y4_nosum_960_new_L.csv", header = FALSE)$V1
y5_asy_cvm <- read.csv("CVM_boot_asym1_new_L.csv", header = FALSE)$V1



setEPS()
postscript("PRD_sphered_cvm_new_LL1.eps", width = 9, height = 6)
par(mar = c(4, 5.5, 2, 2))
make_plot(list(y1_cvm, y2_cvm, y3_cvm, y4_cvm), ordering = c(1,3,2,4), xlim=c(0.05,1.6),cex = 2, 
          ylab = expression(P(C[N]<=c[N])),xlab=expression(c[N]))
dev.off()

setEPS()
postscript("PRD_rotated_cvm_new_LL.eps",width = 9, height = 6)
par(mar = c(4, 5.5, 2, 2))
make_plot(list(y1_rot_cvm, y2_rot_cvm, y3_rot_cvm, y4_rot_cvm, y5_asy_cvm), xlim=c(0.01,0.12),
          ordering = c(1,3,2,4,5), cex = 2, ylab = expression(P(tilde(C)[N]<=tilde(c)[N])),xlab=expression(tilde(c)[N]))
dev.off()

