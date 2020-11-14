#devtools::install_github("meysubb/cfbscrapR")
library(cfbscrapR)
library(tidyverse)

cfb_get_drive_pred_pts <- function(year, type = "both", w = NULL, t = NULL, game_id = NULL, pbp = NULL){
  
  kickoffs <- c('Kickoff Return (Offense)', 'Timeout', 'Kickoff', 'Kickoff Return Touchdown')
  
  if(is.null(pbp)){
    games <- cfb_pbp_data(year, season_type = type, week = w, team = t, epa_wpa = TRUE)
  }
  
  else{
    games <- pbp
  }
  
  if(!is.null(game_id)){
    games <- games %>% filter(game_id == game_id)
  }
  
  games <- games %>% mutate(
    succ = ifelse(down == 1, ifelse(yards_gained >= 0.4*distance, 1, 0),
                  ifelse(down == 2, ifelse(yards_gained >= 0.6*distance, 1, 0), 
                         ifelse(yards_gained >= distance, 1, 0))
    )) %>% 
    filter(!(play_type %in% kickoffs))
  
  drives <- games %>% 
    filter(!(play_type %in% kickoffs)) %>% 
    group_by(game_id, drive_number) %>% 
    summarise(
      off = offense_play,
      def = defense_play,
      csr = mean(succ, na.rm = T),
      tepa = sum(EPA, na.rm = T),
      pc = sum(EPA, na.rm = T)*csr,
      start = drive_start_yards_to_goal,
      score_diff = score_diff_start,
      start_quarter = first(period),
      mean_epa = mean(EPA, na.rm = T),
      num_plays = n(),
      std_epa = ifelse(num_plays != 1, sd(EPA, na.rm = T), 0),
      start_time_till_half_end = drive_time_minutes_start*60*ifelse((start_quarter == 1), 2, 1)*ifelse((start_quarter == 3), 2, 1) + drive_time_seconds_start,
      drive_points = ifelse(drive_result == 'PUNT TD', -1*drive_pts, drive_pts), 
      touchdown = ifelse(drive_result == 'TD', 1,
                         ifelse(drive_result == 'END OF HALF TD', 1,
                                ifelse(drive_result == 'FG TD', 1,
                                       ifelse(drive_result == 'PUNT TD', 1,
                                              ifelse(drive_result == 'END OF GAME TD', 1, 0))))),
      fg = ifelse(drive_result == 'FG', 1, 0),
      opp_td = ifelse(drive_result == 'FUMBLE RETURN TD', 1,
                      ifelse(drive_result == 'INT TD', 1,
                             ifelse(drive_result == 'FUMBLE TD', 1,
                                    ifelse(drive_result == 'DOWNS TD', 1,
                                           ifelse(drive_result == 'PUNT RETURN TD', 1,
                                                  ifelse(drive_result == 'MISSED FG TD', 1, 0)))))),
      saf = ifelse(drive_result == 'SF', 1, 0),
      result = drive_result,
      week = week
    ) %>% 
    distinct() %>% 
    filter(start != 0)
  
  drive_td_pred <- readRDS("cfb_drive_td_pred.rds")
  drive_fg_pred <- readRDS("cfb_drive_fg_pred.rds")
  drive_opp_td_pred <- readRDS("cfb_drive_opp_td_pred.rds")
  drive_saf_pred <- readRDS("cfb_drive_saf_pred.rds")
  
  drives$pred_td <- predict(drive_td_pred, newdata = drives, allow.new.levels = TRUE)
  drives$td_prob <- exp(drives$pred_td)/(1+exp(drives$pred_td))
  
  drives$pred_fg <- predict(drive_fg_pred, newdata = drives, allow.new.levels = TRUE)
  drives$fg_prob <- exp(drives$pred_fg)/(1+exp(drives$pred_fg))
  
  drives$pred_opp_td <- predict(drive_opp_td_pred, newdata = drives, allow.new.levels = TRUE)
  drives$opp_td_prob <- exp(drives$pred_opp_td)/(1+exp(drives$pred_opp_td))
  
  drives$pred_saf <- predict(drive_saf_pred, newdata = drives, allow.new.levels = TRUE)
  drives$saf_prob <- exp(drives$pred_saf)/(1+exp(drives$pred_saf))
  
  drives$pred_drive_pts <- 7*drives$td_prob + 3*drives$fg_prob -7*drives$opp_td_prob -2*drives$saf_prob
  
  drives$resid <- drives$drive_points - drives$pred_drive_pts
  
  drives <- drives %>% select(-pred_td, -pred_fg, -pred_opp_td, -pred_saf) %>% 
    mutate(
      sum_probs = td_prob + fg_prob + opp_td_prob + saf_prob
    )
  
  return(drives)
}